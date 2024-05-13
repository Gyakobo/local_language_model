import os


# ANSI escape codes for different colors
class colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'

# Example usage
# print(colors.RED + "This is red text" + colors.RESET)

def list_models_in_folder(folder_path):
    # List all models in the given folder
    models = [ file for file in os.listdir(folder_path) if file.endswith('.pth') ] 
    return models

def load_model(folder_path):
    # Load the model specified by the user
    models = list_models_in_folder(folder_path)

    if not models:
        print(colors.RED + "No models found in the folder." + colors.RESET)
        return None # Not if it's the appropriate way to exit

    print(colors.GREEN + "Models available in the folder:" + colors.RESET)
    for i, model in enumerate(models, start=1):
        print(colors.YELLOW + f"{i}. {model}" + colors.RESET)

    print("")

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
                print(colors.RED + "Invalid choice. Please enter a valid number." + colors.RESET)
        except ValueError:
            print(colors.RED + "Invalid choice. Please enter a valid number." + colors.RESET)

'''
def main():
    folder_path = input("Enter the folder path: ")
    if not os.path.exists(folder_path):
        print(colors.RED + "Folder doesn't exist." + colors.RESET)
        return

    if not os.listdir(folder_path):
        print(colors.RED + "Folder is empty." + colors.RESET)
        return

    model_path = load_model(folder_path)
    if model_path:
        print(colors.BLUE + f"Loading model from {model_path}" + colors.RESET)
        # Load the selected model

if  __name__ == "__main__":
    main()
'''

