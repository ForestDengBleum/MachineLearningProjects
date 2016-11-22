# FaceDetect
臉部偵測 for Windows use Python <br>
使用Python練習OpenCV，順便做個紀錄
* [Demo Video](https://youtu.be/TnnSfnjw6js) - Windows 

## 特色
* 臉部偵測(圖片)
   
## 使用方法
請先確定電腦上已經安裝<b> OpenCV </b>和<b> numpy </b><br>
可參考[如何在Windows安裝OpenCV Python設定](https://github.com/twtrubiks/FaceDetect/tree/master/How%20Install%20OpenCV%20in%20on%20Windows%20for%20Python)<br>

```
python face_detect.py [圖片檔名] [haarcascade.xml]
```

## 執行範例 
``` 
python face_detect.py test1.jpg haarcascade_frontalface_alt.xml
```

## 執行過程
程式執行後，會自行跳出偵測臉部的圖片<br>
![alt tag](http://i.imgur.com/u8m9lUf.jpg)

## 輸出
目錄底下會多出一張名稱為 face_detection.jpg 的圖片 <br>
![alt tag](http://i.imgur.com/evl398U.jpg)

## 備註
程式基本上只做了小小的修改，<br>
我發現<b> haarcascade_frontalface_alt.xml </b>比原作者的<b> haarcascade_frontalface_default.xml </b>有更好的辨識率，<br>
更多的<b> haarcascade </b>可到<b> opencv </b>路徑裡<b> opencv\sources\data\haarcascades </b>取得，<br>
路徑底下還有<b> haarcascade_eye.xml </b>偵測眼睛之類的，大家可以自行摸索，<br>
最後，請注意，這是<b> 臉部偵測 </b>，並不是 <b> 臉部辨識 </b>

## 其他
更多的資訊可參考原作者<br>

Run the code like this:

*python face_detect.py abba.png haarcascade_frontalface_default.xml*

If you want to understand how the code works, the details are here:

https://realpython.com/blog/python/face-recognition-with-python/

## 更多參考
[記錄, OpenCV 學習路徑, (3) 人臉辨識 ](http://gogoprivateryan.blogspot.tw/2015/09/opencv-3-opencv-python-face-recognition.html)<br>
這篇文章介紹的蠻詳細的，包括可以自己訓練一套專屬用途的 Cascade classifier<br>

## 執行環境
* Windows 8.1
* Python 2.7.3
* OpenCV 2.4.12 
* numpy 1.11.0

## License
MIT license
