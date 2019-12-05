.. _issues:

Common issues
=============

Below is a list of common issues encountered while using TensorFlow for objects detection.

Python crashes - TensorFlow GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Cleaning up Nvidia containers (TensorFlow GPU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



ModuleNotFoundError: No module named 'deployment' or No module named 'nets'
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This error occurs when you try to run train.py and you don’t have the PATH and PYTHONPATH environment variables set up correctly. Exit the virtual environment by closing and re-opening the Anaconda Prompt window. Then, issue “activate TF_object_detection” to re-enter the environment, and then issue the commands given in [`Step 3.2`](https://github.com/khushi2091/Tensorflow-Custom-Object-Detection-Tutorial#32-add-libraries-to-pythonpath).

You can use “echo %PATH%” and “echo %PYTHONPATH%” to check the environment variables and make sure they are set up correctly.


Protobuf Compilation error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ImportError: cannot import name 'anchor_generator_pb2'; ImportError: cannot import name 'string_int_label_map_pb2'

Any error related to protobuf files occur when the corresponding preprocessor.proto has not been compiled. Re-run the protoc command given in [`Step 3.1`](https://github.com/khushi2091/Tensorflow-Custom-Object-Detection-Tutorial#31-protobuf-compilation). Check the ```C:\TF_object_detection\models\research\object_detection\protos``` folder to make sure there is a name_pb2.py file for every file_name.proto file.
