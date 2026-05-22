from paddleocr import PaddleOCRVL
ocr = PaddleOCRVL(device="cpu")   # use gpu:0 later if paddle cuda works
result = ocr.predict("./tests/test_files/eq4.jpg")
print(type(result))
print(result)