import preprocessing as pp

def __main__():

    with open("C:\\Users\\maxim\\Downloads\\female_abstracts.txt", "r",encoding="utf-8") as f:
        text = f.read()

        # do preprocessing on text by calling method in the preprocessing python file
        processed_text = pp.do_preprocessing(text)


    f.close()
    print("process finished")

if __name__ == "__main__":
    __main__()

