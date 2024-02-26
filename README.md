# YOLOv7-官方文件執行說明檔  
## 建立YOLOv7 所需要的環境以及資料  
1. 先開啟Anaconda Prompt視窗，進入你要的資料夾指令 cd + 資料夾位置  
2. 如何git clone YOLOv7 的官方文件指指令: git clone https://github.com/WongKinYiu/yolov7  
![克隆過程圖片](https://github.com/wangbosen123/YOLOv7-/blob/main/image.png)  
完成以上將可以把官方所需要的檔案全部下載至目標檔案。  
3. 然後進入到項目文件夾中，進行環境的安裝，分別輸入兩行代碼即可，同樣，速度受服務器影響，但都不會太久，指令:  
   1.cd yolov7  
   2.pip install -r requirements.txt  
![環境安裝](https://github.com/wangbosen123/YOLOv7-/blob/main/%E7%92%B0%E5%A2%83%E5%AE%89%E8%A3%9D.png)
## 下載YOLOv7 官方預訓練的模型參數  
5. 接著要建立一個文件夾，這個文件夾要裝官方預訓練過的yolov7模型，**預先訓練過的模型權重有較好的初始(預先訓練很重要)**，接續任務中可以模型可以較快的收斂以及較佳的結果，指令:  
   1. mkdir weights  
   2. cd weights  
   ![建立權重的資料夾](https://github.com/wangbosen123/YOLOv7-/blob/main/%E5%BB%BA%E7%AB%8B%E6%AC%8A%E9%87%8D%E8%B3%87%E6%96%99%E9%9B%86.png)
6. 在官方github 下載預先訓練的模型權重至建立好的weights資料夾底下  
   網址1: https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt  
   網址2: https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt  
   ![下載預先訓練模型的權重](https://github.com/wangbosen123/YOLOv7-/blob/main/%E4%B8%8B%E8%BC%89%E9%A0%90%E5%85%88%E8%A8%93%E7%B7%B4%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%AC%8A%E9%87%8D.png)
   
## 嘗試用YOLOv7官方提供的預訓練模型先對資料進行偵測  
8. 嘗試用下載下來的權重對官方給的圖像進行測試，其中  
   --weights weight/yolov7.pt   # 這個參數是把已經訓練好的模型路徑傳進去，就是前面下載的模型權重  
   --source inference/images   # 傳進去要偵測的圖片 
   --save-txt   #偵測圖片bounding box 座標  
   ![執行測試](https://github.com/wangbosen123/YOLOv7-/blob/main/%E5%9F%B7%E8%A1%8C%E6%B8%AC%E8%A9%A6.png)
   ### 測試結果(有加入--save-txt 才會有labels的資料夾)  
   ![測試結果](https://github.com/wangbosen123/YOLOv7-/blob/main/%E6%B8%AC%E8%A9%A6%E7%B5%90%E6%9E%9C.png)

## 建立YOLOv7 所需要的資料集  
資料集的編排方式如下:  
![資料格式](https://github.com/wangbosen123/YOLOv7-/blob/main/dataform.png)  
程式碼在prepare_data.py  


## training前需要先建立pytorch環境
pytorch-gpu 安裝:  
"cuda, cudnn 安裝方式":  
https://blog.csdn.net/weixin_42496865/article/details/124002488  

"pytorch, torchvision, torchaudio 安裝方式":  
https://pytorch.org/get-started/previous-versions/  
命令為:pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113  
"可能遇到多線程的問題"(訓練模型時如果遇到workers的問題加入以下程式碼在train.py裡面主程式的部分便可以解決):  
解決方式為在主程式前面加上，os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  
如果遇到其他沒有安裝的套件就pip install 目標套件  這樣及可以順利安裝  

## training 前必須先配置模型文件  
第一步: 複製yolov7.yaml 文件到相同的路徑下，命名成你想要偵測專案的名稱。  
第二步: 打開以上文件，進行下圖所示的修改，這裡修改的地方只有一處，就是把nc修改為我們資料集的目標總數集可，然後儲存。  
![執行測試](https://github.com/wangbosen123/YOLOv7-/blob/main/%E9%85%8D%E7%BD%AE%E6%A8%A1%E5%9E%8B%E6%96%87%E4%BB%B6.png)  

## training 前必須先配置資料集文件  
第一步: 複製coco.yaml檔案到相同的路徑下，命名成你想要偵測專案的名稱。  
第二步: 打開以上文件，進行下圖所示的修改，需要修改的地方為5處。 第一個：把程式碼自動下載COCO資料集的指令註解掉，以防程式碼自動下載資料集佔用記憶體；第二處：修改train的位置為train_list.txt的路徑；第三處：修改val的位置為 val_list.txt的路徑；第四處：修改nc為資料集目標總數；第五處：修改names為資料集所有目標的名稱。 然後儲存。  
![執行測試](https://github.com/wangbosen123/YOLOv7-/blob/main/%E9%85%8D%E7%BD%AE%E8%B3%87%E6%96%99%E9%9B%86%E6%96%87%E4%BB%B6.png)  

## Training 細節
在 Anaconda Prompt 上面執行 train 的指令:  
python train.py --weights weights/yolov7_training.pt --cfg cfg/training/yolov7-Helmet.yaml --data data/Helmet.yaml --device 0,1 --batch-size 64 --epoch 10  
--weights 代表我們取用官方預訓練(Pre-train)的模型當作我們模型一開始的初始，再用我們的資料對模型微調(fine-tune)  
--cfg 讀取有關訓練細節相關資訊的文件  
--data 讀取有關訓練資料相關資訊的文件  
--device 取用顯示卡推薦使用一個就好，使用兩個電腦會過熱當機  
-- batch-size 批次的數量，需要能整除的訓練總數  
-- epoch 所有資料經過模型訓練的總次數  
以下在yolov7 train.py 裡面還有許多參數可以設定例如:  
![訓練參數設定](https://github.com/wangbosen123/YOLOv7-/blob/main/train_parameters.png)  
在train 結束後，模型會自動儲存最好一次epoch的模型權重，以及各個階段模型的權重，也存了其他的文件，如下圖所示:  

## Training 數據圖大致講解  
### 1.首先要先介紹在YOLOv7內部的cofunsion matrix 他跟我們平常理解的不太一樣，  
左上角代表真實標籤是腫瘤，預測出來也為腫瘤的機率。  
左下角代表真實標籤是腫瘤，預測出來沒有東西(代表背景)的機率。  
右上角代表在背景中有框出腫瘤，只有有框出來這邊的數字就為1，因為總共被框出來的背景有n張，同時被框出腫瘤也為n張。  
右下角代表真實是背景卻預測腫瘤，同時也代表真實是腫瘤但卻沒有預測腫瘤，兩者無交集機率為0。  
如下圖所示:  
![confusion matrix](https://github.com/wangbosen123/YOLOv7-/blob/main/confusion_matrix.png)  

## F1 score  
F1分數，他被定義為Precision以及Recall的調和平均數，最大為1，最小為0，以以下圖為例，可以得知在confidence score = 0.603，則模型在validation datasets 上有較好的成績。  
![F1 score](https://github.com/wangbosen123/YOLOv7-/blob/main/f1_score.png)  

## loss curve  
這邊的損失函數曲線圖是針對訓練以及測試呈現，其中  
box: 回歸損失，針對預測的bounding box以及正確答案的bounding box的差異。  
objectness: 推測偵測框中對物體偵測有無的分類損失。  
classification: 種類的分類損失，因為我們做的是單分類，所以沒有這個損失。  
![loss curve](https://github.com/wangbosen123/YOLOv7-/blob/main/loss_curve.png)   

## PR curve  
PR 曲線中P代表的precision，R代表的是Recall，他們倆著之間是一個矛盾的性能指標，及一個值越高，另一個值越低;  
提高Precision(TP/(TP+FP)) >> 則FP越小  
提高Recall(TP/(TP+FN)) >> 則FN越小  
![PR curve](https://github.com/wangbosen123/YOLOv7-/blob/main/PR_curve.png)  

## val prediction  
左圖為test label 右圖為最後模型針對testing data 做的預測。  
![Test prediction](https://github.com/wangbosen123/YOLOv7-/blob/main/test_prediction.png)  


## inference (detect.py) 參數說明  
前面有說明到執行inference 的指令這邊多介紹幾個之後會調整到的參數:  
--weights以及--source 是前面介紹過的  
--img-size 代表你要測試的圖像你需要到縮小放大的尺寸。  
--conf-thres 可以根據f1 score 的最高點來進行設置。  
--save-txt 之後都必須加上這個，因為需要將預測的座標點儲存起來方便後續評估。  
![detect parameters](https://github.com/wangbosen123/YOLOv7-/blob/main/detect%20parameter.png)  




