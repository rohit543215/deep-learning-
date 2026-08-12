test_loss,test_accuracy=model.evaluate(x_test,y_test)
print(f"test loss:{test_loss} and test_accuracy:{test_accuracy}")

predictions=model.predict(x_test[:10])
predicted_digits=np.argmax(predictions,axis=1)
true_digits=np.argmax(y_test[:10],axis=1)

fig,axes=plt.subplots(2,5,figsize=(12,6))
for i,ax in enumerate(axes.flat):
    ax.imshow(x_test[i].reshape(28,28),cmap='gray')
    color='green' if predicted_digits[i]==true_digits[i] else 'red'
    ax.set_title(f"pred : {predicted_digits[i]}\nTrue: {true_digits[i]}", color=color)
    ax.axis('off')
plt.suptitle('mnist predictions', fontsize=14)
plt.tight_layout()
plt.show()    