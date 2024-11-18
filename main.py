from configparser import ConfigParser
import TrainPipeline
import TestPipeline
import ExplainPipeline

def main():

    options = ["TRAIN", "TEST", "EXPLAIN"]

    cfg_parser = ConfigParser()
    cfg_parser.read("config.conf")
    
    print("Options available: " ) 
    for i, option in enumerate(options, start=1):
        print(f"{i}: {option}")
    
    try:
        choice = int(input("Choose (1-3): "))
        assert choice in range(1, len(options) + 1)
    except AssertionError:
        print("Invalid choice")
        return

    match choice:
        case 1:
            conf = cfg_parser["TRAIN"]
            TrainPipeline.execute_pipeline(conf)
        case 2:
            conf = cfg_parser["TEST"]
            TestPipeline.execute_pipeline(conf)
        case 3:
            conf = cfg_parser["EXPLAIN"]
            ExplainPipeline.execute_pipeline(conf)
        case _:
            return

if __name__ == "__main__":
    main()