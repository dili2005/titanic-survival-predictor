#IMPORT THE LIBRARIES 
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#LOAD THE DATA 
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

#DROP THE COLUMNS THAT ARE USELESS
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

##FILL MISSING AGE WITH AVARAGE AGE 
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Fill missing Embarked with most common port
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])


#TRANCELATE WORD INTO NUMBERS
#male = 0, femal = 1
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Fill missing Embarked with most common port
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])


#SPLIT INTO FEATURES AND TARGET
#x = everything network see
#y = what it's predict 
X = df.drop(columns=['Survived']).values
y = df['Survived'].values

#SCALE FEATURE TO SAME RANGE
scaler = StandardScaler()
X = scaler.fit_transform(X)

#SLPIT 80% TRAIN AND 20% TEST 
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#CONVERT TO TENSORS
X_train = torch.tensor(X_train,dtype=torch.float32)
X_test = torch.tensor(X_test,dtype=torch.float32)
y_train = torch.tensor(y_train,dtype=torch.float32)
y_test = torch.tensor(y_test,dtype=torch.float32)

# Create datasets and loaders
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


#BUILD THE NURAL NETWORK 
class TitanicNet(nn.Module):
    def __init__(self):
        super(TitanicNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(7,64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32,1),
            nn.Sigmoid()
        )
        
        
    def  forward(self,x):
        return self.network(x)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TitanicNet().to(device)

#LOSS FUNCTION 
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005)

#TRAIING LOOP 
epochs = 100

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for X_batch,y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        output = model(X_batch).squeeze()
        loss = criterion(output,y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print("Epoch " + str(epoch + 1) + " Loss: " + str(round(total_loss / len(train_loader), 4)))
         
         
model.eval()
with torch.no_grad():
    X_test_gpu = X_test.to(device)
    output = model(X_test_gpu).squeeze()
    predictions = (output >= 0.5).float()
    y_test_gpu = y_test.to(device)
    accuracy = (predictions == y_test_gpu).float().mean()
    print("Test Accuracy: " + str(round(accuracy.item() * 100, 2)) + "%")
    
    
torch.save(model.state_dict(),'titanic_model.pth')
print("model saved !")