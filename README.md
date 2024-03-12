# README

### Purpose

- Encapsulation of implementation of empirical models
- Brute force search of best model configuration
- Graphic presentation of training history

### Example of training and prediction

    from blackboxes.box import Black
    phi = Black()
    y = phi(X=X, Y=Y, x=x, backend='keras', neurons=[6,4], trainer='adam')
