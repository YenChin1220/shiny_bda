library(shiny)
library(reticulate)
setwd("/Users/yenchin/Documents/BDA/shiny01")

use_python("/opt/anaconda3/bin/python")
source_python("BigDataAnalysis_Package.py")

scaler_paths <- list(
  Xa = "Xa_scale.pkl",
  Xb = "Xb_scale.pkl",
  Ya = "Ya_scale.pkl",
  Yb = "Yb_scale.pkl"
)

model_paths <- list(
  Xa_dis = "Xa_dis.pth",
  Xa_con = "Xa_con.pth",
  Xb_dis = "Xb_dis.pth",
  Xb_con = "Xb_con.pth",
  Ya_dis = "Ya_dis.pth",
  Ya_con = "Ya_con.pth",
  Yb_dis = "Yb_dis.pth",
  Yb_con = "Yb_con.pth"
)

