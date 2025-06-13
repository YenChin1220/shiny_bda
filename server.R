server <- function(input, output, session) {
  observe({
    if (input$model == "rf") {
      disable("task")  # 讓 radioButtons 不能點選
    } else {
      enable("task")   # 如果選 cnn，就重新啟用
    }
  })
  observeEvent(input$predict, {
    req(input$file)
    
    file_path <- input$file$datapath
    group <- input$group
    label <- ifelse(group %in% c("Xa", "Xb"), "X", "Y")
    
    # CNN 模型：使用 1D 單軸時序資料
    if (input$model == "cnn") {
      scale <- load_scaler(scaler_paths[[group]])
      model_path <- model_paths[[paste0(group, "_", input$task)]]
      
      pred <- if (input$task == "con") {
        predict_con(file_path, scale, model_path = model_path, label = label, folder = FALSE)
      } else {
        predict_dis(file_path, scale, model_path = model_path, label = label, folder = FALSE)
      }
      
      output$info <- renderText({ paste("檔案屬於：", group) })
      output$info2 <- renderText({ paste("模型：", ifelse(input$task == "con", "連續型預測", "離散型預測")) })
      output$result <- renderPrint({
        if (input$task == "con") {
          cat("連續型預測結果： ",pred)
        }else{
          cat("離散型預測結果： ",pred)
        }
      })
      
      output$result2 <- renderUI({
        reference <- if (label == "X") c(65, 80, 95, 130) else c(220, 260, 300, 380)
        closest <- reference[which.min(abs(reference - pred))]
        is_normal <- (label == "X" && closest == 80) || (label == "Y" && closest == 260)
        color <- if (is_normal) "green" else "red"
        comment <- if (is_normal) "正常負荷" else "不正常"
        
        HTML(paste0(
          "<span style='color:", color, "; font-size:16px;'>",
          "預測類別：", closest, " → ", comment, "</span><br>",
          if (!is_normal && input$task == "dis" &&
              ((label == "X" && pred == 1) || (label == "Y" && pred == 1))) {
            "<span style='color:orange;'>⚠️ 雖然預測為健康，但類別為異常負載，請留意。</span>"
          }
        ))
      })
    }
    #RF
    else if (input$model == "rf") {
      
      result <- tryCatch({
        input_json <- jsonlite::toJSON(list(file_path = file_path), auto_unbox = TRUE)
        system2("python3", args = "rf_model_wrapper.py", input = input_json, stdout = TRUE)
      }, error = function(e) {
        return(paste0('{"error": "', e$message, '"}'))
      })
      
      if (length(result) == 0 || any(grepl("error", result, ignore.case = TRUE))) {
        showNotification("⚠️ 模型未能正確輸出，請確認資料格式與模型狀態", type = "error")
        return()
      }
      
      res <- jsonlite::fromJSON(result)
      
      output$info <- renderText({ paste("檔案屬於：", input$group) })
      
      output$result <- renderUI({
        req(res)
        prob <- res$Health_Probability
        
        if (prob >= 70) {
          label <- "健康"
          color <- "green"
        } else if (prob >= 45) {
          label <- "注意"
          color <- "orange"
        } else {
          label <- "不健康"
          color <- "red"
        }
        
        HTML(paste0(
          "<b><span style='color:", color, "; font-size:15px;'>",
          "健康預測：", label,"</span></b><br>",#"（機率：", prob, "%）
          "<span style='font-size:15px;'>預測負載：", res$Predicted_Load, "</span>"
        ))
      })
      
      output$result2 <- renderUI({
        req(res)
        reference <- if (label == "X") c(65, 80, 95, 130) else c(220, 260, 300, 380)
        closest <- reference[which.min(abs(reference - res$Predicted_Load))]
        is_normal <- (label == "X" && closest == 80) || (label == "Y" && closest == 260)
        color <- if (is_normal) "green" else "red"
        comment <- if (is_normal) "正常負荷" else "不正常"
        
        HTML(paste0(
          "️<span style='color:", color, "; font-size:15px;'>",
          "最接近類別：", closest, " → ", comment,
          "</span><br>",
          if (!is_normal && res$Health_Prediction ==1) {
            "<span style='color:orange;'>⚠️ 雖然模型預測為健康，但負載值為異常區間，請留意。</span>"
          }
        ))
      })
    }
    
  })
}
