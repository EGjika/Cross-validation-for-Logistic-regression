library(shiny)
library(caret)
library(glmnet)
library(ggplot2)

# Define UI
ui <- fluidPage(
  titlePanel("K-Fold Cross-Validation for Logistic Regression"),
  
  sidebarLayout(
    sidebarPanel(
      fileInput("datafile", "Upload CSV File", accept = ".csv"),
      uiOutput("var_select_ui"),
      numericInput("num_k_values", "Max Number of Folds (k)", value = 10, min = 2),
      actionButton("run_cv", "Run Cross-Validation")
    ),
    
    mainPanel(
      verbatimTextOutput("model_summary"),
      tableOutput("cv_results_table"),
      plotOutput("cv_results_plot")
    )
  )
)

# Define server
server <- function(input, output, session) {
  # Reactive to load uploaded data
  dataset <- reactive({
    req(input$datafile)
    read.csv(input$datafile$datapath)
  })
  
  # Dynamically render variable selectors
  output$var_select_ui <- renderUI({
    req(dataset())
    tagList(
      selectInput("target_var", "Select Target Variable (Binary)", choices = names(dataset())),
      selectInput("predictors", "Select Predictor Variables", choices = names(dataset()), multiple = TRUE)
    )
  })
  
  # Perform k-fold cross-validation
  perform_cv <- function(data, target, predictors, k_values) {
    # Ensure the target is a factor for binary classification
    y <- as.factor(data[[target]])  
    
    # Select only numeric predictors
    x <- data[, predictors, drop = FALSE]
    x <- x[, sapply(x, is.numeric)]  # Filter out non-numeric columns
    
    # Impute missing values (mean imputation)
    pre_process <- preProcess(x, method = "medianImpute")  # Impute NAs with median
    x <- predict(pre_process, newdata = x)  # Apply imputation
    
    mean_accuracies <- numeric(length(k_values))
    std_devs <- numeric(length(k_values))
    
    for (i in seq_along(k_values)) {
      folds <- createFolds(y, k = k_values[i])
      fold_accuracies <- numeric(k_values[i])
      
      for (j in seq_along(folds)) {
        # Split into training and validation sets
        validation_indices <- folds[[j]]
        x_train <- scale(x[-validation_indices, ])
        y_train <- y[-validation_indices]
        x_val <- scale(x[validation_indices, ],
                       center = attr(x_train, "scaled:center"),
                       scale = attr(x_train, "scaled:scale"))
        y_val <- y[validation_indices]
        
        # Train logistic regression model
        model <- glmnet(as.matrix(x_train), y_train, family = "binomial")
        
        # Predict on validation data
        predictions <- predict(model, as.matrix(x_val), type = "response")
        predicted_classes <- ifelse(predictions > 0.5, levels(y)[2], levels(y)[1])
        
        # Calculate accuracy
        fold_accuracies[j] <- mean(predicted_classes == y_val)
      }
      
      # Store mean accuracy and standard deviation for the current fold count (k)
      mean_accuracies[i] <- round(mean(fold_accuracies), 3)
      std_devs[i] <- round(sd(fold_accuracies), 3)
    }
    
    return(data.frame(k = k_values, Mean_Accuracy = mean_accuracies, Std_Dev = std_devs))
  }
  
  # Reactive for cross-validation results
  cv_results <- reactive({
    req(input$run_cv, dataset(), input$target_var, input$predictors)
    data <- dataset()
    k_values <- seq(2, input$num_k_values, by = 1)
    perform_cv(data, input$target_var, input$predictors, k_values)
  })
  
  # Render model summary
  output$model_summary <- renderPrint({
    req(input$run_cv, dataset(), input$target_var, input$predictors)
    data <- dataset()
    glm(as.formula(paste(input$target_var, "~", paste(input$predictors, collapse = "+"))),
        data = data, family = "binomial")
  })
  
  # Render cross-validation table
  output$cv_results_table <- renderTable({
    req(cv_results())
    cv_results()
  })
  
  # Render cross-validation plot
  output$cv_results_plot <- renderPlot({
    req(cv_results())
    results <- cv_results()
    ggplot(results, aes(x = k, y = Mean_Accuracy)) +
      geom_line(color = "blue") +
      geom_point(color = "blue", size = 2) +
      geom_errorbar(aes(ymin = Mean_Accuracy - Std_Dev, ymax = Mean_Accuracy + Std_Dev),
                    width = 0.2, color = "red") +
      labs(title = "Cross-Validation Results", x = "Number of Folds (k)", y = "Mean Accuracy") +
      theme_minimal()
  })
}

# Run the app
shinyApp(ui = ui, server = server)
