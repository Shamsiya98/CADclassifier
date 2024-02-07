import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os 
import cv2
from tqdm import tqdm
from PIL import Image
from keras.utils import to_categorical


image_dir=r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\train"

bed_d=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\train\BED-DOUBLE")
bed_s=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\train\BED-SINGLE")
dw=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\train\DISHWASHER")
door_d=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\train\DOOR-DOUBLE")
door_s=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\train\DOOR-SINGLE")
door_w=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\train\DOOR-WINDOWED")
refg=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\train\REFRIGERATOR")
shower=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\train\SHOWER")
sink=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\train\SINK")
sofa_c=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\train\SOFA-CORNER")
sofa_one=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\train\SOFA-ONE")
sofa_three=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\train\SOFA-THREE")
sofa_two=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\train\SOFA-TWO")
stove_ov=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\train\STOVE-OVEN")
table_dinner=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\train\TABLE-DINNER")
table_stdy=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\train\TABLE-STUDY")
tv=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\train\TELEVISION")
toilet=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\train\TOILET")
wardrobe=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\train\WARDROBE")
washbasin=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\train\WASHBASIN")
washbasin_cab=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\train\WASHBASIN-CABINET")
washing_mach=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\train\WASHINGMACHINE")
window=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\train\WINDOW")

dataset=[]
label=[]
img_size=(112,112)

for i , image_name in tqdm(enumerate(bed_d),desc="bed_double"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/BED-DOUBLE/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(0)

for i , image_name in tqdm(enumerate(bed_s),desc="bed_single"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/BED-SINGLE/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(1)
        

for i , image_name in tqdm(enumerate(dw),desc="dishwasher"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/DISHWASHER/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(2)
        

for i , image_name in tqdm(enumerate(door_d),desc="door_double"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/DOOR-DOUBLE/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(3)
        

for i , image_name in tqdm(enumerate(door_s),desc="door_single"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/DOOR-SINGLE/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(4)
        

for i , image_name in tqdm(enumerate(door_w),desc="door_windowed"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/DOOR-WINDOWED/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(5)
      

for i , image_name in tqdm(enumerate(refg),desc="refrigerator"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/REFRIGERATOR/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(6)
    

for i , image_name in tqdm(enumerate(shower),desc="shower"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/SHOWER/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(7)
     

for i , image_name in tqdm(enumerate(sink),desc="sink"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/SINK/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(8)
       

for i , image_name in tqdm(enumerate(sofa_c),desc="sofa_corner"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/SOFA-CORNER/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(9)
        

for i , image_name in tqdm(enumerate(sofa_one),desc="sofa_one"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/SOFA-ONE/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(10)

for i , image_name in tqdm(enumerate(sofa_three),desc="sofa_three"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/SOFA-THREE/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(11)
        

for i , image_name in tqdm(enumerate(sofa_two),desc="sofa_two"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/SOFA-TWO/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(12)
     

for i , image_name in tqdm(enumerate(stove_ov),desc="stove_ov"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/STOVE-OVEN/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(13)
    

for i , image_name in tqdm(enumerate(table_dinner),desc="table_dinner"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/TABLE-DINNER/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(14)
   

for i , image_name in tqdm(enumerate(table_stdy),desc="table_stdy"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/TABLE-STUDY/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(15)
       

for i , image_name in tqdm(enumerate(tv),desc="tv"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/TELEVISION/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(16)
        

for i , image_name in tqdm(enumerate(toilet),desc="toilet"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/TOILET/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(17)
      

for i , image_name in tqdm(enumerate(wardrobe),desc="wardrobe"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/WARDROBE/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(18)
     

for i , image_name in tqdm(enumerate(washbasin),desc="washbasin"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/WASHBASIN/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(19)
 

for i , image_name in tqdm(enumerate(washbasin_cab),desc="washbasin_cab"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/WASHBASIN-CABINET/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(20)
   

for i , image_name in tqdm(enumerate(washing_mach),desc="washing_mach"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/WASHINGMACHINE/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(21)
  

for i , image_name in tqdm(enumerate(window),desc="window"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/WINDOW/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(22)

dataset=np.array(dataset)
label = to_categorical(label, num_classes=23)
label = np.array(label)

print('Dataset Length: ',len(dataset))
print('Label Length: ',len(label))

print("Normalaising the Dataset. \n")

dataset = dataset.astype('float')/255.0

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(112, 112, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(23, activation='softmax')  
])

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)
history=model.fit(dataset, label, epochs=10, batch_size =128,validation_split=0.1,callbacks=[early_stop])
model.save("model.h5")

print("-------------------------MODEL TESTING PHASE---------------------------------")

image_dir=r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\test"

bed_d=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\test\BED-DOUBLE")
bed_s=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\test\BED-SINGLE")
dw=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\test\DISHWASHER")
door_d=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\test\DOOR-DOUBLE")
door_s=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\test\DOOR-SINGLE")
door_w=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\test\DOOR-WINDOWED")
refg=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\test\REFRIGERATOR")
shower=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\test\SHOWER")
sink=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\test\SINK")
sofa_c=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\test\SOFA-CORNER")
sofa_one=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\test\SOFA-ONE")
sofa_three=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\test\SOFA-THREE")
sofa_two=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\test\SOFA-TWO")
stove_ov=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\test\STOVE-OVEN")
table_dinner=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\test\TABLE-DINNER")
table_stdy=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\test\TABLE-STUDY")
tv=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\test\TELEVISION")
toilet=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\test\TOILET")
wardrobe=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\test\WARDROBE")
washbasin=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\test\WASHBASIN")
washbasin_cab=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\test\WASHBASIN-CABINET")
washing_mach=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\test\WASHINGMACHINE")
window=os.listdir(r"C:\Users\HP\Desktop\CAD CLASSIFICATION\FDS\test\WINDOW")

dataset=[]
label=[]
img_size=(112,112)

for i , image_name in tqdm(enumerate(bed_d),desc="bed_double"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/BED-DOUBLE/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(0)

for i , image_name in tqdm(enumerate(bed_s),desc="bed_single"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/BED-SINGLE/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(1)

for i , image_name in tqdm(enumerate(dw),desc="dishwasher"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/DISHWASHER/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(2)

for i , image_name in tqdm(enumerate(door_d),desc="door_double"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/DOOR-DOUBLE/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(3)

for i , image_name in tqdm(enumerate(door_s),desc="door_single"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/DOOR-SINGLE/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(4)

for i , image_name in tqdm(enumerate(door_w),desc="door_windowed"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/DOOR-WINDOWED/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(5)

for i , image_name in tqdm(enumerate(refg),desc="refrigerator"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/REFRIGERATOR/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(6)

for i , image_name in tqdm(enumerate(shower),desc="shower"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/SHOWER/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(7)

for i , image_name in tqdm(enumerate(sink),desc="sink"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/SINK/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(8)

for i , image_name in tqdm(enumerate(sofa_c),desc="sofa_corner"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/SOFA-CORNER/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(9)

for i , image_name in tqdm(enumerate(sofa_one),desc="sofa_one"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/SOFA-ONE/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(10)

for i , image_name in tqdm(enumerate(sofa_three),desc="sofa_three"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/SOFA-THREE/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(11)

for i , image_name in tqdm(enumerate(sofa_two),desc="sofa_two"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/SOFA-TWO/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(12)

for i , image_name in tqdm(enumerate(stove_ov),desc="stove_ov"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/STOVE-OVEN/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(13)

for i , image_name in tqdm(enumerate(table_dinner),desc="table_dinner"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/TABLE-DINNER/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(14)

for i , image_name in tqdm(enumerate(table_stdy),desc="table_stdy"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/TABLE-STUDY/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(15)

for i , image_name in tqdm(enumerate(tv),desc="tv"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/TELEVISION/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(16)

for i , image_name in tqdm(enumerate(toilet),desc="toilet"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/TOILET/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(17)

for i , image_name in tqdm(enumerate(wardrobe),desc="wardrobe"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/WARDROBE/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(18)

for i , image_name in tqdm(enumerate(washbasin),desc="washbasin"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/WASHBASIN/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(19)

for i , image_name in tqdm(enumerate(washbasin_cab),desc="washbasin_cab"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/WASHBASIN-CABINET/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(20)

for i , image_name in tqdm(enumerate(washing_mach),desc="washing_mach"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/WASHINGMACHINE/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(21)

for i , image_name in tqdm(enumerate(window),desc="window"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/WINDOW/'+image_name, cv2.IMREAD_GRAYSCALE)
        image=Image.fromarray(image,'L')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(22)

dataset=np.array(dataset)
label = to_categorical(label, num_classes=23)
label = np.array(label)

print('Dataset Length: ',len(dataset))
print('Label Length: ',len(label))

print("Normalaising the Dataset. \n")

dataset = dataset.astype('float')/255.0

result=model.evaluate(dataset,label)
print(result)