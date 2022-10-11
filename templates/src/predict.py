import argparse
import configparser
from datetime import datetime
import os
import json
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import shutil
import sys
import time
import traceback
import yaml

from logger import Logger

SHOW_LOG = True


class Predictor():

    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config.read("config.ini")
        #self.parser = argparse.ArgumentParser(description="Predictor")
        self.X_train = pd.read_csv(self.config["SPLIT_DATA"]["X_train"], index_col=0)
        self.y_train = pd.read_csv(self.config["SPLIT_DATA"]["y_train"], index_col=0)
        self.X_test = pd.read_csv(self.config["SPLIT_DATA"]["X_test"], index_col = 0)
        self.y_test = pd.read_csv(self.config["SPLIT_DATA"]["y_test"], index_col=0)
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.transform(self.X_test)
        self.log.info("Predictor is ready")

    def predict(self, name:str) -> bool:
        try:
            classifier = pickle.load(open(self.config[name]["path"], "rb"))
        except FileNotFoundError:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        try:
            print(classifier.score(self.X_test,self.y_test))
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        self.log.info(f'{self.config[name]["path"]} is up to update')
        return True

if __name__ == "__main__":
    predictor = Predictor()
    predictor.predict("LOG_REG")