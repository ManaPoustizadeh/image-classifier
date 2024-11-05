# Step-by-Step  Forward Pass Example

Initial Input
* Shape: (1, 1, 32, 32)
* Explanation: This represents one grayscale image with a height and width of 32 pixels.

## First Convolution Layer (`conv1`)

Operation: Applies a 5x5 convolution with 6 filters and ReLU activation.
* self.conv1 = nn.Conv2d(1, 6, 5)
* Input Channels: 1, Output Channels: 6, Kernel Size: 5x5

```
Output Width = 32−5+1 = 28
Output Height = 32−5+1 = 28
```
* Output Shape: (1, 6, 28, 28)

## First Max Pooling Layer (`s2`)

`s2 = F.max_pool2d(c1, (2, 2))`

Operation: Applies a 2x2 max pooling operation, which reduces the spatial dimensions by half.
```
Output Width = 28/2 = 14
Output Height = 28/2 = 14
```
* Output Shape: (1, 6, 14, 14)

## Second Convolutional Layer (`conv2`)

`c3 = F.relu(self.conv2(s2))`

Operation: Applies a 5x5 convolution with 16 filters and ReLU activation.

```
Output Width = 14−5+1 = 10
Output Height = 14−5+1 = 10
```

* Output Shape: (1, 16, 10, 10)

## Second Max Pooling Layer (`s4`)

`s4 = F.max_pool2d(c3, 2)`

Operation: Applies a 2x2 max pooling operation, which again reduces spatial dimensions by half.

```
Output Width = 10/2 = 5
Output Height = 10/2 = 5
```

* Output Shape: (1, 16, 5, 5)

## Flatten Layer

`16 x 5 x 5 = 400`

* Output Shape: (1, 400)

##  First Fully Connected Layer (`fc1`)

Operation: Applies a fully connected layer with 400 inputs and 120 outputs, followed by ReLU activation.

* Output Shape: (1, 120)

## Second Fully Connected Layer (`fc2`)

Operation: Applies a fully connected layer with 120 inputs and 84 outputs, followed by ReLU activation.

`self.fc2 = nn.Linear(120, 84)`

* Output Shape: (1, 84)

##  Final Fully Connected Layer (fc3)

Operation: Applies a fully connected layer with 84 inputs and 10 outputs.

`self.fc3 = nn.Linear(84, 10)`

* Output Shape: (1, 10)

# Summary of Shapes through Each Layer
| Layer                          | Shape             |
|--------------------------------|-------------------|
| Initial Input                  | `(1, 1, 32, 32)` |
| `conv1`                        | `(1, 6, 28, 28)` |
| `max_pool2d` (after `conv1`)   | `(1, 6, 14, 14)` |
| `conv2`                        | `(1, 16, 10, 10)`|
| `max_pool2d` (after `conv2`)   | `(1, 16, 5, 5)`  |
| Flatten                        | `(1, 400)`       |
| `fc1`                          | `(1, 120)`       |
| `fc2`                          | `(1, 84)`        |
| `fc3`                          | `(1, 10)`        |
