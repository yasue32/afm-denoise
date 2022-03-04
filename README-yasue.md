# 引き継ぎ用README
「連続撮影されたSPM画像による相互補完的なノイズ除去」
2021年度課題研究　B4安江楓真


## 1. 内容
ノイズ除去を動かすためのメモ

## 2. 動作環境
主要なライブラリは以下のバージョンで動作確認．
少なくともdl10, dl11, dl20, dl30, dl40で動作した．

[SIFT-flow](https://github.com/hmorimitsu/sift-flow-gpu) 216e230

```
Python == 3.6.9
torch == 1.8.2+cu111
torchvision == 0.9.2
pandas == 1.1.5
numpy == 1.19.5
wheel == 0.37.0
opencv-python == 4.5.4.60
scikit-image == 0.17.2
```


## 3. 主要なファイル
- `iSeeBetterTrain.py` / 学習用．引数については後述．
- `iSeeBetterTest.py` / テスト用．引数については後述．
- `dataset_akita.py` / データローダー．（`dataset.py`は現在使えない．）
- `make_dataset_akita.py` / 連番画像からデータセットを作成．コード内に引数がある．（その他の`make_dataset*.py`は現在使えない．）
- `command_memo.sh` / 実行コマンドの例．再現実験する際のコマンドもあり．

## 4. データセット
- orig_img: オリジナルの画像．ただし明らかにノイズが多すぎる画像はすでに省かれている．
- train_imgs: 学習用画像．データセットはafm_dataset_per_sequence．
- test_imgs: テスト用画像．データセットはtest_dataset_per_sequence．
- ext_clean_img: ノイズが多い画像をさらに目視で省いた学習用画像．データセットはext_clean_dataset_per_sequence．
- _alignがついているデータセットは，アフィン変換による位置合わせがされたもの．

## 5. 学習時やテスト時の引数
### 重要な引数
- ノイズ除去時には，`--denoise`をつける．これをつけると出力画像が超解像サイズにならない．また，入力画像をダウンスケーリングしない．
- 生成器と識別器による学習（GAN）をする場合には`--APITLoss`をつける．識別器の学習をせず生成器のみ学習する場合には，`--RBPN_only`をつける．
- モデルの保存場所は`--save_folder pass/to/save`で指定．
  - 事前学習されたモデルを使う場合，`--pretrained --pretrained_sr model.pth`．
  - この場合実際に読み込もうとするファイルは`pass/to/save/model.pth`になるので注意．
- GPUの指定は引数では行わず，`CUDA_VISIBLE_DEVICES=x,y python3 iSeeBetter...`で行っていた．
  - ２枚以上使うときは`--gpus 2 --useDataParallel`とする．
  - 他にいい方法があれば教えてください．

### その他の使える引数
- `--upscale_factor 2` / 超解像の倍率．事前学習されたモデルを使う場合はそれに合わせる．
- `--batchSize 12` / ミニバッチサイズ．
- `--start_epoch` / 学習を途中から再開する場合などに指定．
- `--nEpochs` / 学習エポック数．
- `--snapshots` / モデルを保存する頻度．
- `--lr` / 学習率
- `--gpu_mode` / GPUを使用する場合True．
- `--threads` / データローダのスレッド数．データローダが結構重いので，ボトルネックになりそうなら多め推奨．
- `--seed` / シード値．
- `--data_dir` / データセットのディレクトリ．
- `--nFrames` / 入力する画像数．
- `--patch_size` / 学習時のクロップサイズ．学習時は64で，テスト時は256にするなどもできる．
- `--model_type` / RBPNか，比較用のUNetを指定．
- `--residuals` / 残差ネットワーク．
- `--debug` / ほぼ効果なし．
- `--log_dir` / tensorboardで保存するディレクトリ
- `--warping` / 画像の位置合わせ．warpingはオプティカルフローによるピクセルごとの位置合わせ，alignmentはアフィン変換による画像全体の位置合わせを行う．
- `--use_wandb --use_tensorboard` / W&Bやtensorboardを導入してログの記録を行う場合．
- `--num_channels` / 入力画像のチャンネル数
- `--optical_flow` / オプティカルフローを入力するか否か．n=無し，p=pyflowを入力．
- `--pretrained_d` / 識別器の事前学習モデルが有る場合に指定．
- `--noise_flow_type` / ノイズ定量化の手法の選択．p=フィルターベースの手法．s=オプティカルフローによる手法．
- `--aloss` / GANを学習する際の，最高性誤差に対するadversarial lossの重み
### テスト時のみの引数
- `upscale_only` / 入力画像をダウンスケーリングしない場合．
- `--all_patarn` / テスト時に入力できるパターンを全て試す．例えば9枚1組の画像から，入力画像を7枚選ぶ場合，$_9C_7=36$通りを全て行う．出力が膨大になるので少ないデータセットで行うことを推奨．

### 現在は使えない引数
- `--testBatchSize  --gpu_id  --file_list --other_dataset --future_frame --data_augmentation　--prefix --shuffle --depth_img` 
- `--alignment`は使えないが，アフィン変換で位置合わせしたい場合は`make_dataset_akita.py`でデータセット作成する際に予め位置合わせしておく．


## 6. その他
- iSeeBetterはRBPNの時系列超解像にadversarial lossなどの損失関数を組み込んだものである．
