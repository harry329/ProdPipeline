# Put the code for your API here.
from starter.ml.data import clean_data
from starter.train_model import train_save_model, make_prediction

if __name__ == "__main__":
    clean_data()
    train_save_model()
    make_prediction()
