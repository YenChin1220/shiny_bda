library(shiny)
library(shinythemes)
library(shinyjs)  

ui <- fluidPage(
  useShinyjs(), 
  tags$head(
    tags$link(rel = "stylesheet", type = "text/css", href = "style.css")
  ),
  div(class = "custom-container",

      div(class = "custom-title", "震動預測小幫手"),
      div(style = "max-width: 500px; margin: 40px auto 0 auto;",
          
          wellPanel(
            selectInput("model", "模型選擇：", choices = c("CNN" = "cnn", "Random Forest" = "rf")),
            fileInput("file", "上傳測試檔 (.txt)", accept = ".txt"),
            selectInput("group", "資料來源：",
                        choices = c("Xa - 水平傳動軸馬達側" = "Xa",
                                    "Xb - 水平傳動軸惰輪側" = "Xb",
                                    "Ya - 垂直傳動軸馬達側" = "Ya",
                                    "Yb - 垂直傳動軸惰輪側" = "Yb")),
            options = list(
              dropdownParent = 'body',
              highlight = TRUE),
            radioButtons("task", "預測任務：", choices = c("連續型" = "con", "離散型" = "dis"), 
                         selected = "con", inline = TRUE),
            actionButton("predict", "開始預測", class = "btn btn-primary")
          ),
          
          br(),

          wellPanel(
            textOutput("info"),
            conditionalPanel(
              condition = "input.model == 'cnn'",
              textOutput("info2")
            ),
            
            htmlOutput("result"),
            htmlOutput("result2")
          )
          
      )
  )
)
