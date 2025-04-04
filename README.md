
# **Einops from scratch**
This repository contains the code for implementation of rearrange function for tensors that support rearangement and repetition transformations. 

## **Contents**
1. [Submission details](#submission-details)
2. [Running the Code and Tests](#running-the-code-and-tests)  
   - [Clone the Repository](#clone-the-repository)  
   - [Install Dependencies](#install-dependencies)  
   - [Running the Code](#running-the-code)  
   - [Running Tests](#running-tests)  
2. [Implementation approach](#implementation-approach)  
   - [Overview of the Process](#overview-of-the-process)  
   - [Detailed Steps](#detailed-steps)  
     - [Parser](#1-parser)  
     - [Information Extractor](#2-information-extractor)  
     - [Dimension Calculator](#3-dimension-calculator)  
     - [Transformer](#4-transformer)  
   - [High-Level Flow Diagram](#high-level-flow-diagram)  

---
# Submission details

Todo: provide notebook examples

# **Running the Code and Tests**

## **Clone the Repository**  
```bash
git clone <repository_url>
cd <repository_name>
```

## **Install Dependencies**  
```bash
pip install .
```

---

## **Running the Code**  

Once installed, you can use the `rearrange` function inside Python:

```python
from einops_rearrange import rearrange
import numpy as np

tensor = np.random.rand(2, 3, 4)
pattern = "b h w -> h w b"

result = rearrange(tensor, pattern)
print(result.shape)  # Expected output: (3, 4, 2)
```

---

## **Running Tests**  

All unit tests are located in the `tests/` directory.  
Execute the following command from the project root to run all tests:

```bash
pytest tests/
```

---

# **Implementation approach**

Following is the description of the `rearrange` function, which supports `rearrange` and `repeat` functionality from `einops`. The module applies tensor transformations based on a pattern string, just like `einops`. The approach is broken into four distinct steps:

1. **Parser**
2. **Information Extractor**
3. **Dimension Calculator**
4. **Transformer**

---

## **Overview of the Process**

The overall workflow for the `rearrange` function is as follows:

1. **Input Validation:**  
   The main function does input checks.

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

---

## **Detailed Steps**

### **1. Parser**
- **Module:** `parser.py`  
- **Purpose:**  
  - Breaks down the provided pattern into a structured format.
  - Validates the presence and proper use of tokens like ellipsis (`...`), numeric dimensions (anonymous axes), and named axes.  

---

### **2. Information Extractor**
- **Module:** `transform.py` (function: `extract_information`)  
- **Purpose:**  
  - Processes the parsed input and output patterns (obtained by splitting the pattern with `"->"`).  
  - Accounts for ellipsis by dynamically generating axis names if needed.  
  - Establishes mappings for axis-to-position and their respective lengths.  
  - Determines known and unknown dimensions, as well as the permutation order for transposition.  

---

### **3. Dimension Calculator**
- **Module:** `transform.py` (function: `infer_shape`)  
- **Purpose:**  
  - Computes the necessary reshaping parameters before and after the main transformation.  
  - Infers unknown axis sizes by comparing known products of dimensions with the actual tensor shape.  
  - Determines whether an initial or final reshape is required based on the pattern structure.  

---

### **4. Transformer**
- **Module:** `transform.py` (function: `apply_transform`)  
- **Purpose:**  
  - Applies the computed transformations to the tensor using numpy `reshape`, `transpose`, and `expand_dims`, based on the data provided by previous functions.  
  - **Initial Reshape:** If the inferred shape indicates, the tensor is reshaped to an intermediate form.  
  - **Transposition:** The tensor’s dimensions are reordered based on the permutation order extracted earlier.  
  - **Repetition:** New axes are inserted and then broadcast to the appropriate size.  
  - **Final Reshape:** If required, the tensor is reshaped into its final target dimensions.  

---

## **High-Level Flow Diagram**

The design is highly modular to ensure isolation, maintainability, and easier unit testing.

Below is a simple diagram illustrating the four-step process:

```mermaid
flowchart TD
    A[Input Tensor & Pattern]
    B[Parser]
    C[Information Extractor]
    D[Dimension Calculator]
    E[Transformer]
    F[Output Tensor]
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
```
