{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "94466f8d-f665-4b0a-bca8-25bb9b0bb7fc",
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "from tensorflow.keras import layers, models\n",
    "from tensorflow.keras.applications import VGG16\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "dbe82084-22c3-48b9-b821-ca455710f151",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_dir = \"C:\\\\Users\\\\Sneha\\\\Downloads\\\\OCT2017\\\\OCT2017\\\\train\"\n",
    "test_dir  = \"C:\\\\Users\\\\Sneha\\\\Downloads\\\\OCT2017\\\\OCT2017\\\\test\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "d4170cfd-8b00-4df0-ad27-7983f5fd365a",
   "metadata": {},
   "outputs": [],
   "source": [
    "IMG_SIZE = (224, 224)\n",
    "batch_size = 32"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "fa54dd51-1ed1-4081-92b5-bf9d220f9a0a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 83484 images belonging to 4 classes.\n"
     ]
    }
   ],
   "source": [
    "train_gen = ImageDataGenerator(\n",
    "    rescale=1./255,\n",
    "    rotation_range=10,\n",
    "    zoom_range=0.1,\n",
    "    horizontal_flip=True,\n",
    "    brightness_range=(0.8,1.2)\n",
    ").flow_from_directory(train_dir, target_size=IMG_SIZE, class_mode='categorical',\n",
    "                       batch_size=batch_size, shuffle=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "7ec7def1-7ad8-4d57-b5fb-b648d017a846",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 1000 images belonging to 4 classes.\n"
     ]
    }
   ],
   "source": [
    "test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(\n",
    "    test_dir, target_size=IMG_SIZE, class_mode='categorical', batch_size=batch_size)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "56f49b99-0afc-433e-b8c2-1dd20db02769",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5\n",
      "\u001b[1m58889256/58889256\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m8s\u001b[0m 0us/step\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\">Model: \"sequential\"</span>\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1mModel: \"sequential\"\u001b[0m\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n",
       "┃<span style=\"font-weight: bold\"> Layer (type)                    </span>┃<span style=\"font-weight: bold\"> Output Shape           </span>┃<span style=\"font-weight: bold\">       Param # </span>┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n",
       "│ vgg16 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Functional</span>)              │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">7</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">7</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)      │    <span style=\"color: #00af00; text-decoration-color: #00af00\">14,714,688</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ global_average_pooling2d        │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)            │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "│ (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">GlobalAveragePooling2D</span>)        │                        │               │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dense (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                   │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)            │       <span style=\"color: #00af00; text-decoration-color: #00af00\">131,328</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dropout (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dropout</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)            │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dense_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                 │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4</span>)              │         <span style=\"color: #00af00; text-decoration-color: #00af00\">1,028</span> │\n",
       "└─────────────────────────────────┴────────────────────────┴───────────────┘\n",
       "</pre>\n"
      ],
      "text/plain": [
       "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n",
       "┃\u001b[1m \u001b[0m\u001b[1mLayer (type)                   \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1mOutput Shape          \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1m      Param #\u001b[0m\u001b[1m \u001b[0m┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n",
       "│ vgg16 (\u001b[38;5;33mFunctional\u001b[0m)              │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m7\u001b[0m, \u001b[38;5;34m7\u001b[0m, \u001b[38;5;34m512\u001b[0m)      │    \u001b[38;5;34m14,714,688\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ global_average_pooling2d        │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m512\u001b[0m)            │             \u001b[38;5;34m0\u001b[0m │\n",
       "│ (\u001b[38;5;33mGlobalAveragePooling2D\u001b[0m)        │                        │               │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dense (\u001b[38;5;33mDense\u001b[0m)                   │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m256\u001b[0m)            │       \u001b[38;5;34m131,328\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dropout (\u001b[38;5;33mDropout\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m256\u001b[0m)            │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dense_1 (\u001b[38;5;33mDense\u001b[0m)                 │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m4\u001b[0m)              │         \u001b[38;5;34m1,028\u001b[0m │\n",
       "└─────────────────────────────────┴────────────────────────┴───────────────┘\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Total params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">14,847,044</span> (56.64 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Total params: \u001b[0m\u001b[38;5;34m14,847,044\u001b[0m (56.64 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">132,356</span> (517.02 KB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Trainable params: \u001b[0m\u001b[38;5;34m132,356\u001b[0m (517.02 KB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Non-trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">14,714,688</span> (56.13 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Non-trainable params: \u001b[0m\u001b[38;5;34m14,714,688\u001b[0m (56.13 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# 3️⃣ Build model with transfer learning\n",
    "base_model = VGG16(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))\n",
    "base_model.trainable = False\n",
    "\n",
    "model = models.Sequential([\n",
    "    base_model,\n",
    "    layers.GlobalAveragePooling2D(),\n",
    "    layers.Dense(256, activation='relu'),\n",
    "    layers.Dropout(0.3),\n",
    "    layers.Dense(train_gen.num_classes, activation='softmax')\n",
    "])\n",
    "\n",
    "model.compile(optimizer='adam',\n",
    "              loss='categorical_crossentropy',\n",
    "              metrics=['accuracy'])\n",
    "\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "4029bae3-6d1b-4bec-921c-d727b8bbef73",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Sneha\\anaconda3\\Lib\\site-packages\\keras\\src\\trainers\\data_adapters\\py_dataset_adapter.py:121: UserWarning: Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.\n",
      "  self._warn_if_super_not_called()\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10\n",
      "\u001b[1m2609/2609\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m3758s\u001b[0m 1s/step - accuracy: 0.7351 - loss: 0.7105\n",
      "Epoch 2/10\n",
      "\u001b[1m2609/2609\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m22055s\u001b[0m 8s/step - accuracy: 0.8319 - loss: 0.4563\n",
      "Epoch 3/10\n",
      "\u001b[1m2609/2609\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m20570s\u001b[0m 8s/step - accuracy: 0.8389 - loss: 0.4339\n",
      "Epoch 4/10\n",
      "\u001b[1m2609/2609\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m36525s\u001b[0m 14s/step - accuracy: 0.8465 - loss: 0.4180\n",
      "Epoch 5/10\n",
      "\u001b[1m2609/2609\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m4046s\u001b[0m 2s/step - accuracy: 0.8521 - loss: 0.4029\n",
      "Epoch 6/10\n",
      "\u001b[1m2609/2609\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m3782s\u001b[0m 1s/step - accuracy: 0.8529 - loss: 0.4000\n",
      "Epoch 7/10\n",
      "\u001b[1m2609/2609\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m13819s\u001b[0m 5s/step - accuracy: 0.8574 - loss: 0.3878\n",
      "Epoch 8/10\n",
      "\u001b[1m2609/2609\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m18548s\u001b[0m 7s/step - accuracy: 0.8571 - loss: 0.3907\n",
      "Epoch 9/10\n",
      "\u001b[1m2609/2609\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m6356s\u001b[0m 2s/step - accuracy: 0.8591 - loss: 0.3845\n",
      "Epoch 10/10\n",
      "\u001b[1m2609/2609\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m41973s\u001b[0m 16s/step - accuracy: 0.8634 - loss: 0.3736\n"
     ]
    }
   ],
   "source": [
    "# 4️⃣ Train the model\n",
    "history = model.fit(\n",
    "    train_gen,\n",
    "    epochs=10\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "cbd0029b-e5ac-44dd-a31e-953c67a89caa",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m32/32\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m39s\u001b[0m 1s/step - accuracy: 0.9119 - loss: 0.2568\n",
      "✅ Test Accuracy: 91.30%\n"
     ]
    }
   ],
   "source": [
    "# 5️⃣ Evaluate & visualize\n",
    "test_loss, test_acc = model.evaluate(test_gen)\n",
    "print(f\"✅ Test Accuracy: {test_acc*100:.2f}%\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
