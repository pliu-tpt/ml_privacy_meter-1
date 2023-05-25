import pickle
import sys

"""
Small script that reads the metadata in demo
"""
if __name__ == "__main__":
    folder = sys.argv[1]
    with open(f'./{folder}/models_metadata.pkl', 'rb') as f:
        data = pickle.load(f)
        for model in data["model_metadata"].keys():
            print(25 * "#")
            print(f"{model}-th Model")
            print(data["model_metadata"][model])
