import statistics.preprocessing as pp
import statistics.feature_extraction as fe

def __main__():

    with open("female.txt", "r",encoding="utf-8") as f:
        
        text = f.read()
        # do preprocessing on text by calling method in the preprocessing python file
        processed_text = pp.do_preprocessing(text)
        #documents = processed_text.split('\n')

        # do feature extraction on text by calling method in the feature_extraction python file
        #feature = fe.bag_of_words(documents)
        #print(feature)

    f.close()

    with open("male.txt", "r",encoding="utf-8") as f:
        
        text = f.read()
        # do preprocessing on text by calling method in the preprocessing python file
        processed_text = pp.do_preprocessing(text)
        #documents = processed_text.split('\n')

        # do feature extraction on text by calling method in the feature_extraction python file
        #feature = fe.bag_of_words(documents)
        #print(feature)

    f.close()

    print("process finished")

if __name__ == "__main__":
    __main__()

