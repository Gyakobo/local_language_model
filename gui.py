import os

def list_models_in_folder(folder_path):
    # List all models in the given folder
    models = [ file for file in os.listdir(folder_path) if file.endswith('.pth') ] 
    return models

def load_model(folder_path):
    # Load the model specified by the user
    models = list_models_in_folder(folder_path)

    if not models:
        print("No models found in the folder.")
        return None # Not if it's the appropriate way to exit

    print("Models available in the folder:")
    for i, model in enumerate(models, start=1):
        print(f"{i}. {model}")

    print("\n")

    while True:
        choice = input("Enter the number of the model you want to load (or 'exit' to quit): ")
        if choice.lower() == 'exit':
            return None # Still unsure about this 
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(models):
                model_path = os.path.join(folder_path, models[choice_idx])
                return model_path
            else:
                print("Invalid choice. Please enter a valid number.")
        except ValueError:
            print("Invalid choice. Please enter a valid number.")

def main():
    folder_path = input("Enter the folder path: ")
    if not os.path.exists(folder_path):
        print("Folder doesn't exist.")
        return

    if not os.listdir(folder_path):
        print("Folder is empty.")
        return

    model_path = load_model(folder_path)
    if model_path:
        print(f"Loading model from {model_path}")
        # Load the selected model

if  __name__ == "__main__":
    main()

