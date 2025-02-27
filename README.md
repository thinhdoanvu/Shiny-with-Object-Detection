## Install libraries
${\textsf{\color{blue}reticulate allows calling Python from R}}$
```
install.packages("shiny")
install.packages("reticulate")
```

## Make Virtual Environment (open in terminal)
```
python -m venv shiny_yolov8
```

## Active shiny-yolov8
```
source shiny_yolov8/bin/activate  # Windows: myenv\Scripts\activate
```

## Install yolov8 and components
```
pip install ultralytics opencv-python numpy
```

## Configure reticulate in R (app.R file)
```
library(reticulate)
use_virtualenv("~/Users/thinhdoanvu/.virtualenvs/shiny_yolov8", required = TRUE)

```
## app.R 
```
library(shiny)
library(reticulate)

# Cấu hình môi trường Python
options(shiny.maxRequestSize = 500 * 1024^2)  # Giới hạn upload 500MB

use_virtualenv("/Users/thinhdoanvu/shiny_yolov8/")

# Load YOLO từ Python
py_run_string("
from ultralytics import YOLO
import torch
import cv2
import numpy as np

model = None

def load_model(model_path):
    global model
    model = YOLO(model_path)

def predict(image_path):
    if model is None:
        return 'Model chưa được load!', None
    
    results = model(image_path)
    detections = len(results[0].boxes)
    
    img = results[0].plot()
    output_path = image_path.replace('.jpg', '_output.jpg').replace('.png', '_output.png')
    cv2.imwrite(output_path, img)
    
    if detections > 0:
        return f'{detections} đối tượng được phát hiện.', output_path
    else:
        return 'Không phát hiện đối tượng nào.', output_path
")

# Giao diện Shiny
ui <- fluidPage(
  titlePanel("YOLOv8 Object Detection"),
  
  sidebarLayout(
    sidebarPanel(
      fileInput("model", "Upload YOLOv8 Model (.pt)", accept = ".pt"),
      fileInput("image", "Upload Image", accept = c("image/png", "image/jpeg")),
      actionButton("detect", "Detect"),
      textOutput("result_text")
    ),
    
    mainPanel(
      fluidRow(
        column(6, h3("Ảnh gốc"), imageOutput("input_img", width = "100%", height = "auto")),
        column(6, h3("Ảnh nhận dạng"), imageOutput("output_img", width = "100%", height = "auto"))
      )
    )
  )
)

# Server logic
server <- function(input, output, session) {
  
  observeEvent(input$model, {
    req(input$model)
    model_path <- input$model$datapath
    py$load_model(model_path)
  })
  
  observeEvent(input$image, {
    req(input$image)
    output$input_img <- renderImage({
      list(src = input$image$datapath, contentType = "image/png", width = "100%", height = "auto")
    }, deleteFile = FALSE)
  })
  
  observeEvent(input$detect, {
    req(input$image)
    img_path <- input$image$datapath
    
    result <- py$predict(img_path)
    
    output$result_text <- renderText({ result[[1]] })
    
    output$output_img <- renderImage({
      list(src = result[[2]], contentType = "image/png", width = "100%", height = "auto")
    }, deleteFile = FALSE)
  })
}

shinyApp(ui, server)

```

<img width="909" alt="image" src="https://github.com/user-attachments/assets/5c417c7d-aae4-4947-a5ea-51392802eb41" />
