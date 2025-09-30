#!/usr/bin/env python3
"""
台灣美食 CNN 模型架構
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import TopKCategoricalAccuracy
import matplotlib.pyplot as plt

class TaiwanFoodCNN:
    """台灣美食 CNN 模型"""
    
    def __init__(self, num_classes=101, img_size=(224, 224), model_type='custom'):
        """
        初始化 CNN 模型
        
        Args:
            num_classes: 類別數量
            img_size: 輸入圖片大小
            model_type: 模型類型 ('custom', 'resnet50', 'efficientnet', 'mobilenet')
        """
        self.num_classes = num_classes
        self.img_size = img_size
        self.model_type = model_type
        self.model = None
        
    def build_custom_cnn(self):
        """建立自定義 CNN 模型"""
        model = models.Sequential([
            # 第一組卷積層
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # 第二組卷積層
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # 第三組卷積層
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # 第四組卷積層
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # 全域平均池化
            layers.GlobalAveragePooling2D(),
            
            # 全連接層
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            
            # 輸出層
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_transfer_learning_model(self, base_model_name='resnet50'):
        """建立遷移學習模型"""
        
        # 選擇基礎模型
        if base_model_name == 'resnet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        elif base_model_name == 'efficientnet':
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        elif base_model_name == 'mobilenet':
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        else:
            raise ValueError(f"不支援的基礎模型: {base_model_name}")
        
        # 凍結基礎模型的權重
        base_model.trainable = False
        
        # 建立完整模型
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_model(self):
        """根據指定類型建立模型"""
        if self.model_type == 'custom':
            self.model = self.build_custom_cnn()
        elif self.model_type in ['resnet50', 'efficientnet', 'mobilenet']:
            self.model = self.build_transfer_learning_model(self.model_type)
        else:
            raise ValueError(f"不支援的模型類型: {self.model_type}")
        
        return self.model
    
    def compile_model(self, learning_rate=0.001, optimizer='adam'):
        """編譯模型"""
        if self.model is None:
            raise ValueError("請先建立模型")
        
        # 選擇優化器
        if optimizer == 'adam':
            opt = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            opt = optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy', TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
        )
        
        return self.model
    
    def get_callbacks(self, model_name='taiwan_food_cnn'):
        """獲取訓練回調函數"""
        callbacks = [
            # 模型檢查點
            ModelCheckpoint(
                filepath=f'models/{model_name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            
            # 早停
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # 學習率調整
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, train_generator, val_generator, epochs=50, callbacks=None):
        """訓練模型"""
        if self.model is None:
            raise ValueError("請先建立並編譯模型")
        
        if callbacks is None:
            callbacks = self.get_callbacks()
        
        # 建立模型目錄
        import os
        os.makedirs('models', exist_ok=True)
        
        # 開始訓練
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def fine_tune(self, train_generator, val_generator, epochs=10, learning_rate=1e-5):
        """微調模型（僅適用於遷移學習模型）"""
        if self.model_type == 'custom':
            print("自定義模型不支援微調")
            return None
        
        # 解凍基礎模型的部分層進行微調
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # 只微調最後幾層
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # 重新編譯，使用較低的學習率
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
        )
        
        # 微調訓練
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=self.get_callbacks('taiwan_food_cnn_finetuned'),
            verbose=1
        )
        
        return history
    
    def plot_training_history(self, history):
        """繪製訓練歷程"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 準確率
        axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 損失
        axes[0, 1].plot(history.history['loss'], label='Training Loss')
        axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Top-5 準確率
        if 'top_5_accuracy' in history.history:
            axes[1, 0].plot(history.history['top_5_accuracy'], label='Training Top-5 Accuracy')
            axes[1, 0].plot(history.history['val_top_5_accuracy'], label='Validation Top-5 Accuracy')
            axes[1, 0].set_title('Model Top-5 Accuracy')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Top-5 Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # 學習率變化
        if 'lr' in history.history:
            axes[1, 1].plot(history.history['lr'], label='Learning Rate')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def get_model_summary(self):
        """獲取模型摘要"""
        if self.model is None:
            return "模型尚未建立"
        
        return self.model.summary()
    
    def save_model(self, filepath):
        """儲存模型"""
        if self.model is None:
            raise ValueError("模型尚未建立")
        
        self.model.save(filepath)
        print(f"模型已儲存至: {filepath}")
    
    def load_model(self, filepath):
        """載入模型"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"模型已從 {filepath} 載入")
        return self.model

# 使用範例
if __name__ == "__main__":
    # 建立不同類型的模型
    models_to_test = ['custom', 'resnet50', 'efficientnet', 'mobilenet']
    
    for model_type in models_to_test:
        print(f"\n=== 建立 {model_type.upper()} 模型 ===")
        
        try:
            # 建立模型
            cnn = TaiwanFoodCNN(num_classes=101, model_type=model_type)
            model = cnn.build_model()
            cnn.compile_model()
            
            print(f"{model_type} 模型建立成功")
            print(f"總參數數量: {model.count_params():,}")
            
            # 顯示模型結構（僅前幾層）
            print("\n模型結構:")
            for i, layer in enumerate(model.layers[:5]):
                print(f"  {i+1}. {layer.__class__.__name__}: {layer.output_shape}")
            if len(model.layers) > 5:
                print(f"  ... 還有 {len(model.layers)-5} 層")
                
        except Exception as e:
            print(f"建立 {model_type} 模型時發生錯誤: {e}")