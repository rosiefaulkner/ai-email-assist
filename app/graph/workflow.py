from typing import Annotated, Any, Dict, List

from langgraph.graph import StateGraph
from langgraph.graph.state import END

from ..agents.gemini_agent import GeminiAgent
from ..agents.rag_agent import RAGAgent

"""
RAGWorkflow: A class that encapsulates the LangGraph workflow for the RAG application.
Attributes:
    rag_agent (RAGAgent): An instance of the RAGAgent class.
    llm_agent (LLMAgent): An instance of the LLMAgent class.
    graph (Graph): The LangGraph workflow.
Methods:
    _build_graph(self) -> Graph: Build the LangGraph workflow.
    _retrieve_context(self, state: Dict[str, Any]) -> Dict[str, Any]: Retrieve relevant context for the query.
    _generate_response(self, state: Dict[str, Any]) -> Dict[str, Any]: Generate response using the LLM.
    _validate_response(self, state: Dict[str, Any]) -> Dict[str, Any]: Validate the generated response.
    _should_regenerate(self, state: Dict[str, Any]) -> bool: Determine if response should be regenerated.
    run(self, inputs: Dict[str, Any]) -> Dict[str, Any]: Run the workflow with given inputs.
"""


class RAGWorkflow:
    def __init__(self):
        self.rag_agent = RAGAgent()
        self.llm_agent = GeminiAgent()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Initialize the graph
        # Update StateGraph initialization to use Annotated for state schema
        workflow = StateGraph(Annotated[Dict, "email_assistant_state"])

        # Define nodes
        workflow.add_node("retrieve", self._retrieve_context)
        workflow.add_node("generate", self._generate_response)
        workflow.add_node("validate", self._validate_response)

        # Define edges
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "validate")
        workflow.add_edge("validate", END)

        # Add conditional edge for response regeneration
        workflow.add_conditional_edges(
            "validate", self._should_regenerate, {True: "generate", False: END}
        )

        # Compile the graph
        workflow.set_entry_point("retrieve")
        workflow.add_edge("validate", END)
        return workflow.compile()

    async def _retrieve_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant context for the query."""
        query = state["query"]
        relevant_docs = await self.rag_agent.retrieve_relevant_documents(query)

        return {**state, "context": relevant_docs, "attempt": 0}

    async def _generate_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response using the LLM."""
        query = state["query"]
        context = state.get("context", [])

        response = await self.rag_agent.process_query(
            query=query, context=[doc["content"] for doc in context]
        )

        return {**state, "response": response, "attempt": state.get("attempt", 0) + 1}

    async def _validate_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the generated response."""
        response = state["response"]

        # Check for error conditions
        if response.get("error"):
            return {**state, "valid": False, "error": response["error"]}

        # Validate response quality
        validation_prompt = f"""Rate the quality of this response on a scale of 0 to 1:
{response['answer']}

Provide only the numerical score."""

        quality_score = await self.llm_agent.generate_response(validation_prompt)
        quality_score = float(quality_score["answer"].strip())

        return {**state, "valid": quality_score >= 0.7, "quality_score": quality_score}

    def _should_regenerate(self, state: Dict[str, Any]) -> bool:
        """Determine if response should be regenerated."""
        return not state.get("valid", False) and state.get("attempt", 0) < 3

    async def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the workflow with given inputs."""
        try:
            result = await self.graph.arun(inputs)

            # Format final response
            response = result["response"]
            return {
                "answer": response["answer"],
                "sources": response.get("sources", []),
                "metadata": {
                    **response.get("metadata", {}),
                    "quality_score": result.get("quality_score"),
                    "attempts": result.get("attempt", 1),
                },
            }

        except Exception as e:
            return {
                "error": str(e),
                "answer": None,
                "sources": [],
                "metadata": {"error_type": type(e).__name__},
            }
