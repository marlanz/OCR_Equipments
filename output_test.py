from paddleocr import PaddleOCRVL
ocr = PaddleOCRVL(device="gpu:0")   # use gpu:0 later if paddle cuda works
result = ocr.predict("./tests/test_files/eq4.jpg")
print(type(result))
print(result)