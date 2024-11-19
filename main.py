from configparser import ConfigParser
import TrainPipeline
import TestPipeline
import ExplainPipeline
from preprocessing import FeatureExtractor

def main():

    options = ["FEATURES_EXTRACTION", "TRAIN", "TEST", "EXPLAIN"]

    cfg_parser = ConfigParser()
    cfg_parser.read("config.conf")
    
    print("Options available: " ) 
    for i, option in enumerate(options, start=0):
        print(f"{i}: {option}")
    
    try:
        choice = int(input("Choose (0-3): "))
        assert choice in range(len(options) + 1)
    except AssertionError:
        print("Invalid choice")
        return

    match choice:
        case 0:
            conf = cfg_parser[options[choice]]
            FeatureExtractor.execute_pipeline(conf)
        case 1:
            conf = cfg_parser[options[choice]]
            TrainPipeline.execute_pipeline(conf)
        case 2:
            conf = cfg_parser[options[choice]]
            TestPipeline.execute_pipeline(conf)
        case 3:
            conf = cfg_parser[options[choice]]
            ExplainPipeline.execute_pipeline(conf)
        case _:
            return

if __name__ == "__main__":
    main()