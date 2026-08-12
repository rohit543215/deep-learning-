x_train=x_train.reshape(-1,28,28,1)
x_test=x_test.reshape(-1,28,28,1)

x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255

y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)

model=Sequential([
     Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
     Conv2D(32,(3,3), activation='relu'),
     MaxPooling2D(pool_size=(2,2)),
     Dropout(0.25),
     Conv2D(64,(3,3),activation='relu'),
     Conv2D(64,(3,3),activation='relu'),
     MaxPooling2D(pool_size=(2,2)),
     Dropout(0.25),
     Flatten(),
     Dense(128,activation='relu'),
     Dropout(0.5),
     Dense(10,activation='softmax')
     



])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping=EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True

)


learning_rate_reduction=ReduceLROnPlateau(
    monitor='val_accuracy',
    patience=3,
    verbose=1,
    factor=0.5,
    min_lr=0.00001
)

checkpointer=ModelCheckpoint(
    filepath='best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

hist=model.fit(
    x_train,y_train,
    batch_size=64,
    epochs=10,
    validation_data=(x_test,y_test),
    callbacks=[early_stopping,learning_rate_reduction,checkpointer]
)