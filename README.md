# sompy
a numpy based SOM (Self Organizing Map) Library


## Description

## Requirement
+ Python2

## Install 
    pip install sompy
    
## Demo
    from sompy import SOM
    import numpy as np
    import matplotlib.pyplot as plt


    input_data = np.random.rand(10000, 3)
    output_shape = (40, 40)
    som = SOM(output_shape, input_data)
    som.set_parameter(neighbor=0.1, learning_rate=0.2)

    output_map = som.train(10000)

    plt.imshow(output_map,
               interpolation='none')
    plt.show()
![alt tag](https://github.com/ttlg/sompy/blob/master/examples/sample_color.png)


## License
[MIT](http://b4b4r07.mit-license.org)

