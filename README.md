# YOLOv7-官方文件執行說明檔  
1. 先開啟Anaconda Prompt視窗，進入你要的資料夾指令 cd + 資料夾位置  
2. 如何git clone YOLOv7 的官方文件指指令: git clone https://github.com/WongKinYiu/yolov7  
![克隆過程圖片](https://github.com/wangbosen123/YOLOv7-/blob/main/image.png)  
完成以上將可以把官方所需要的檔案全部下載至目標檔案。  
3. 然後進入到項目文件夾中，進行環境的安裝，分別輸入兩行代碼即可，同樣，速度受服務器影響，但都不會太久，指令:  1. cd yolov7
   1.cd yolov7  
   2.pip install -r requirements.txt  
![環境安裝](https://github.com/wangbosen123/YOLOv7-/blob/main/%E7%92%B0%E5%A2%83%E5%AE%89%E8%A3%9D.png)  

4. 接著要建立一個文件夾，這個文件夾要裝官方預訓練過的yolov7模型，**預先訓練過的模型權重有較好的初始(預先訓練很重要)**，接續任務中可以模型可以較快的收斂以及較佳的結果，指令:  
   1.mkdir weights
   2. cd weights  
   ![建立權重的資料夾](https://github.com/wangbosen123/YOLOv7-/blob/main/%E5%BB%BA%E7%AB%8B%E6%AC%8A%E9%87%8D%E8%B3%87%E6%96%99%E9%9B%86.png)
5. 在官方github 下載預先訓練的模型權重至建立好的weights資料夾底下  
   網址1: https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt  
   網址2: https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt  
   ![下載預先訓練模型的權重](https://github.com/wangbosen123/YOLOv7-/blob/main/%E4%B8%8B%E8%BC%89%E9%A0%90%E5%85%88%E8%A8%93%E7%B7%B4%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%AC%8A%E9%87%8D.png)
在 Anaconda Prompt 上面執行 train 的指令:   
在 Anaconda Prompt 上面執行 inference 的指令 :  



