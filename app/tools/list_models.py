import google.generativeai as genai
from app.config import get_settings

def list_available_models():
    settings = get_settings()
    genai.configure(api_key=settings.GOOGLE_API_KEY)
    
    try:
        models = genai.list_models()
        print("\nAvailable Models:")
        for model in models:
            print(f"\nName: {model.name}")
            print(f"Display Name: {model.display_name}")
            print(f"Description: {model.description}")
            print(f"Generation Methods: {', '.join(model.supported_generation_methods)}")
            print("-" * 80)
    except Exception as e:
        print(f"Error listing models: {str(e)}")

if __name__ == "__main__":
    list_available_models()