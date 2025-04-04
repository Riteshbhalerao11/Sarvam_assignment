# **Einops from Scratch**

This repository contains the code for the implementation of the `rearrange` function for tensors. It supports rearrangement and repetition transformations. The implementation is optimized using efficient parsing, utilizing only native Python functions and minimizing the number of array transformations.

---

## **Contents**
1. [Submission Details](#submission-details)
2. [Running the Code and Tests](#running-the-code-and-tests)  
   - [Clone the Repository](#clone-the-repository)  
   - [Install Dependencies](#install-package-and-dependencies)  
   - [Running the Code](#running-the-code)  
   - [Running Tests](#running-tests)
3. [Brief Function Documentation](#brief-function-documentation)
4. [Implementation Approach](#implementation-approach)  
   - [Overview of the Process](#overview-of-the-process)  
   - [Detailed Steps](#detailed-steps)  
     - [Parser](#1-parser)  
     - [Information Extractor](#2-information-extractor)  
     - [Dimension Calculator](#3-dimension-calculator)  
     - [Transformer](#4-transformer)  
   - [High-Level Flow Diagram](#high-level-flow-diagram)  
5. [References](#references)  

---

## **Submission Details**

Complete implementation in one notebook is present [here](https://colab.research.google.com/drive/1OGQrqFRkmgbZTz0qJ_OUcWMhukI5xqDz?usp=sharing). It contains the source code, unit tests, and some primitive examples. Alternatively, you can follow the steps in the subsequent section to install the package locally and test it.  

---

## **Running the Code and Tests**

### **Clone the Repository**  
```bash
git clone https://github.com/Riteshbhalerao11/Sarvam_assignment.git
cd Sarvam_assignment
```

### **Install Package and Dependencies**  
```bash
pip install .
```

### **Running the Code**  

Once installed, you can use the `rearrange` function inside Python:

```python
from einops_rearrange import rearrange
import numpy as np

tensor = np.random.rand(2, 3, 4)
pattern = "b h w -> h w b"

result = rearrange(tensor, pattern)
print(result.shape)  # Expected output: (3, 4, 2)
```

### **Running Tests**  

All unit tests are located in the `tests/` directory.  
Execute the following command from the project root to run all tests:

```bash
pytest tests/
```

---

## Brief Function Documentation
### `einops_rearrange.rearrange`

### **Parameters**

| Name          | Type                               | Description                                                                 | Default |
|--------------|----------------------------------|-------------------------------------------------------------------------|---------|
| `tensor`     | `numpy.ndarray`    | A tensor from numpy library. | **Required** |
| `pattern`    | `str`                             | String specifying the desired rearrangement pattern.                     | **Required** |
| `axes_lengths` | `int`                          | Additional specifications for inferred dimensions.                       | `{}` |

### **Returns**

| Type   | Description |
|--------|-------------|
| `Tensor` | Tensor of the same type as the input|

For usage examples click [here](https://colab.research.google.com/drive/1OGQrqFRkmgbZTz0qJ_OUcWMhukI5xqDz#scrollTo=Examples)

**Note**  
- `(axis1 axis2 ..)` are used for composing axes.  
- `()` empty parentheses count as a unitary axis (not the same as an anonymous unit axis of size `1`).  
- `...` ellipsis can be used as a placeholder for multiple axes.  

---

## **Implementation Approach**

Following is the description of the `rearrange` function, which supports `rearrange` and `repeat` functionality from `einops`. The package applies tensor transformations based on a pattern string, just like `einops`. The approach is broken into four distinct steps:

1. **Parser**
2. **Information Extractor**
3. **Dimension Calculator**
4. **Transformer**

### **Overview of the Process**

The overall workflow for the `rearrange` function is as follows:

1. **Input Validation:**  
   The main function performs input checks.

2. **Extract Transformation Details:**  
   The pattern string is processed through a parser that validates and breaks it into components.

3. **Compute Dimension Changes:**  
   The extracted information is used to determine the required reshaping, transposition, and repetition to achieve the target tensor layout.

4. **Apply Transformation:**  
   The tensor is transformed step-by-step:
   - **Initial Reshaping (if needed)**
   - **Transposition:** Reordering dimensions based on a computed permutation.
   - **Repetition:** Expanding axes where necessary.
   - **Final Reshaping:** Producing the final output tensor shape.

### **Detailed Steps**

#### **1. Parser**
- **Module:** `parser.py`  
- **Purpose:**  
  - Breaks down the provided pattern into a structured format.
  - Validates the presence and proper use of tokens like ellipsis (`...`), numeric dimensions (anonymous axes), and named axes.  

#### **2. Information Extractor**
- **Module:** `transform.py` (function: `extract_information`)  
- **Purpose:**  
  - Processes the parsed input and output patterns (obtained by splitting the pattern with `"->"`).  
  - Accounts for ellipsis by dynamically generating axis names if needed.  
  - Establishes mappings for axis-to-position and their respective lengths.  
  - Determines known and unknown dimensions, as well as the permutation order for transposition.  

#### **3. Dimension Calculator**
- **Module:** `transform.py` (function: `infer_shape`)  
- **Purpose:**  
  - Computes the necessary reshaping parameters before and after the main transformation.  
  - Infers unknown axis sizes by comparing known products of dimensions with the actual tensor shape.  
  - Determines whether an initial or final reshape is required based on the pattern structure.  

#### **4. Transformer**
- **Module:** `transform.py` (function: `apply_transform`)  
- **Purpose:**  
  - Applies the computed transformations to the tensor using numpy `reshape`, `transpose`, `broadcast`, and `expand_dims`, based on the data provided by previous functions.  
  - **Initial Reshape:** If the inferred shape indicates, the tensor is reshaped to an intermediate form.  
  - **Transposition:** The tensor’s dimensions are reordered based on the permutation order extracted earlier.  
  - **Repetition:** New axes are inserted and then broadcast to the appropriate size.  
  - **Final Reshape:** If required, the tensor is reshaped into its final target dimensions.  

### **High-Level Flow Diagram**

<p align="center">
  <img src="https://github.com/Riteshbhalerao11/Sarvam_assignment/blob/master/assets/einops_diagram.png" alt="Einops Rearrange Diagram">
</p>

---

## **References**

- Rogozhnikov, Alex. "Einops: Clear and reliable tensor manipulations with Einstein-like notation." International Conference on Learning Representations. 2022. [Paper Link](https://openreview.net/pdf?id=oapKSVM2bcj)

### **AI-Generated Content Assistance**

Generative models (GPT 4-o and Claude 3.7 Sonnet) were used only for:
1. Generating some test cases.
2. Optimizing certain list comprehensions for readability.
3. Minimal code refactoring for enhanced readability (eg. fixing indentations). 


