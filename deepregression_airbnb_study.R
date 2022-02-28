## ----'setup', echo = FALSE, cache = FALSE----------------------------------------------------------
library("knitr")
opts_chunk$set(dev = c('pdf'), 
        fig.align = 'center',
        fig.path='figures/',
        cache = F
        ) 
options(formatR.arrow = TRUE,
        width = 70) 


## ---- include=FALSE--------------------------------------------------------------------------------
library("deepregression")
library("keras")

## --------------------------------------------------------------------------------------------------
# url <- "https://github.com/davidruegamer/airbnb/raw/main/munich_clean_text.RDS"
# destfile <- file.path(getwd(), "munich.RDS")
# download.file(url, destfile, mode = "wb")
airbnb <- readRDS("munich.RDS")
airbnb$days_since_last_review <- as.numeric(difftime(airbnb$date, airbnb$last_review))


## --------------------------------------------------------------------------------------------------
y = log(airbnb$price)


## --------------------------------------------------------------------------------------------------
list_of_formulas = list(
  loc = ~ 1 + te(latitude, longitude, df = 5),
  scale = ~ 1
)


## --------------------------------------------------------------------------------------------------
mod_simple <- deepregression(
  y = y, 
  data = airbnb, 
  list_of_formulas = list_of_formulas,
  list_of_deep_models = NULL
)


## --------------------------------------------------------------------------------------------------
class(mod_simple$model)


## --------------------------------------------------------------------------------------------------
str(mod_simple$init_params$parsed_formulas_contents$loc,1)
sapply(mod_simple$init_params$parsed_formulas_contents$loc, "[[", "term")


## --------------------------------------------------------------------------------------------------
deep_model <- function(x)
{
  x %>% 
    layer_dense(units = 5, activation = "relu", use_bias = FALSE) %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 3, activation = "relu") %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 1, activation = "linear")
}


## --------------------------------------------------------------------------------------------------
options(identify_intercept = TRUE)
mod <- deepregression(
  y = y, 
  data = airbnb,
  list_of_formulas = list(
    location = ~ 1 + beds + s(accommodates, bs = "ps") +
      s(days_since_last_review, bs = "tp") + 
      deep(review_scores_rating, reviews_per_month),
    scale = ~1
  ),
  list_of_deep_models = list(deep = deep_model)
)


## --------------------------------------------------------------------------------------------------
mod %>% fit(
  epochs = 100, 
  verbose = FALSE, 
  view_metrics = FALSE,
  validation_split = 0.2
  )


## --------------------------------------------------------------------------------------------------
fitted_vals <- mod %>% fitted()
cor(fitted_vals, y)


## ---- cv_plot--------------------------------------------------------------------------------------
mod_cv <- deepregression(
  y = y, 
  data = airbnb,
  list_of_formulas = list(
    location = ~ 1 + beds + s(accommodates, bs = "ps") +
      s(days_since_last_review, bs = "tp") + 
      deep(review_scores_rating, reviews_per_month),
    scale = ~1
  ),
  list_of_deep_models = list(deep = deep_model)
)

res_cv <- mod_cv %>% cv(
  plot = FALSE, 
  cv_folds = 3,
  epochs = 100
  )
plot_cv(res_cv)


## --------------------------------------------------------------------------------------------------
coef(mod, type="linear")


## ---- plot_mod, fig.height=4, fig.width=8----------------------------------------------------------
par(mfrow=c(1,2))
plot(mod)


## --------------------------------------------------------------------------------------------------
dist <- mod %>% get_distribution()
str(dist, 1)


## ---- quant_mean_plot------------------------------------------------------------------------------
first_obs_airbnb <- as.data.frame(airbnb)[1,,drop=F]
dist1 <- mod %>% get_distribution(first_obs_airbnb)
meanval <- mod %>% mean(first_obs_airbnb) %>% c
q05 <- mod %>% quant(data = first_obs_airbnb, probs = 0.05) %>% c
q95 <- mod %>% quant(data = first_obs_airbnb, probs = 0.95) %>% c
xseq <- seq(q05-1, q95+1, l=1000)
plot(xseq, sapply(xseq, function(x) c(as.matrix(dist1$prob(x)))), type="l",
     ylab = "density(price)", xlab = "price")
abline(v = c(q05, meanval, q95), col="red", lty=2)


## --------------------------------------------------------------------------------------------------
str(mod_simple$model$trainable_weights, 1)


## --------------------------------------------------------------------------------------------------
lambda <- 0.5
addpen <- function(x) lambda * tf$reduce_sum(tf$abs(mod_simple$model$trainable_weights[[1]]))
mod_simple_pen <- deepregression(
  y = y, 
  data = airbnb, 
  list_of_formulas = list_of_formulas,
  list_of_deep_models = NULL, 
  additional_penalty = addpen
)


## ---- smooth_plot_comparison, fig.height=3, fig.width=8--------------------------------------------
form_df_3 <- list(loc = ~ 1 + s(days_since_last_review, df = 3), scale = ~ 1)
form_df_6 <- list(loc = ~ 1 + s(days_since_last_review, df = 6), scale = ~ 1)
form_df_10 <- list(loc = ~ 1 + s(days_since_last_review, df = 10), scale = ~ 1)
args <- list(y = y, data = airbnb, 
             list_of_deep_models = NULL)

mod_df_low <- do.call("deepregression", c(args, list(list_of_formulas = form_df_3)))
mod_df_med <- do.call("deepregression", c(args, list(list_of_formulas = form_df_6)))
mod_df_max <- do.call("deepregression", c(args, list(list_of_formulas = form_df_10)))

mod_df_low %>% fit(epochs = 1000, early_stopping = TRUE, verbose = FALSE)
mod_df_med %>% fit(epochs = 1000, early_stopping = TRUE, verbose = FALSE)
mod_df_max %>% fit(epochs = 1000, early_stopping = TRUE, verbose = FALSE)

par(mfrow=c(1,3))
plot(mod_df_low)
plot(mod_df_med)
plot(mod_df_max)



## --------------------------------------------------------------------------------------------------
embd_mod <- function(x) x %>%
  layer_embedding(input_dim = 1000,
                  output_dim = 100) %>%
  layer_lambda(f = function(x) k_mean(x, axis = 2)) %>%
  layer_dense(20, activation = "tanh") %>% 
  layer_dropout(0.3) %>% 
  layer_dense(2) 


## --------------------------------------------------------------------------------------------------
mod <- deepregression(
  y = y,
  list_of_formulas = 
    list(
      location = ~ 1,
      scale = ~ 1,
      both = ~ 0 + embd_mod(texts)
    ),
  list_of_deep_models = 
    list(embd_mod = embd_mod), 
  mapping = list(1,2,1:2),
  data = airbnb
)


## --------------------------------------------------------------------------------------------------
embd_mod <- function(x) x %>%
  layer_embedding(input_dim = 1000,
                  output_dim = 100) %>%
  layer_lambda(f = function(x) k_mean(x, axis = 2)) %>%
  layer_dense(20, activation = "tanh") %>% 
  layer_dropout(0.3) %>% 
  layer_dense(1) 

form_lists <- list(
  location = ~ 1 + embd_mod(texts),
  scale = ~ 1
)
  
mod <- deepregression(
  y = y,
  list_of_formulas = form_lists,
  list_of_deep_models = 
    list(embd_mod = embd_mod), 
  data = airbnb
  )


## --------------------------------------------------------------------------------------------------
set.seed(42)
n <- 1000
toyX <- rnorm(n)
toyY <- 2*toyX + rnorm(n)    


## --------------------------------------------------------------------------------------------------
deep_model <- function(x)
{
  x %>% 
    layer_dense(units = 100, activation = "relu", use_bias = FALSE) %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 50, activation = "relu") %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 1, activation = "linear")
}


## --------------------------------------------------------------------------------------------------
forms <- list(loc = ~ -1 + toyX + deep_model(toyX), scale = ~ 1)
args <- list(
  y = toyY, 
  data = data.frame(toyX = toyX), 
  list_of_formulas = forms, 
  list_of_deep_models = list(deep_model = deep_model)
)

w_oz <- orthog_control(orthogonalize = TRUE)
wo_oz <- orthog_control(orthogonalize = FALSE)
  
mod_w_oz <- do.call("deepregression", c(args, list(orthog_options = w_oz)))
mod_wo_oz <- do.call("deepregression", c(args, list(orthog_options = wo_oz)))

mod_w_oz %>% fit(epochs = 1000, early_stopping = TRUE, batch_size = 50, verbose = FALSE)
mod_wo_oz %>% fit(epochs = 1000, early_stopping = TRUE, batch_size = 50, verbose = FALSE)

cbind(
  with = c(coef(mod_w_oz, which_param = 1)[[1]]),
  without = c(coef(mod_wo_oz, which_param = 1)[[1]]),
  linmod = coef(lm(toyY ~ 0 + toyX))
)


## --------------------------------------------------------------------------------------------------
function(x){ 
  do.call(your_tfd_dist, 
    lapply(1:ncol(x)[[1]],
      function(i) your_trafo_list_on_inputs[[i]](
        x[,i,drop=FALSE])
    )
  )
}


## --------------------------------------------------------------------------------------------------
toyXinDisguise <- toyX


## --------------------------------------------------------------------------------------------------
form_known <- list(loc = ~ -1 + toyX + deep_model(toyX), scale = ~ 1)
form_unknown <- list(loc = ~ -1 + toyX + deep_model(toyXinDisguise), scale = ~ 1)
form_manual <- list(loc = ~ -1 + toyX + deep_model(toyXinDisguise) %OZ% (toyXinDisguise), scale = ~ 1)
args <- list(
  y = toyY, 
  data = data.frame(toyX = toyX, toyXinDisguise = toyXinDisguise), 
  list_of_deep_models = list(deep_model = deep_model)
)

mod_known <- do.call("deepregression", c(args, list(list_of_formulas = form_known)))
mod_unknown <- do.call("deepregression", c(args, list(list_of_formulas = form_unknown)))
mod_manual <- do.call("deepregression", c(args, list(list_of_formulas = form_manual)))

mod_known %>% fit(epochs = 1000, early_stopping = FALSE, verbose = FALSE)
mod_unknown %>% fit(epochs = 1000, early_stopping = FALSE, verbose = FALSE)
mod_manual %>% fit(epochs = 1000, early_stopping = FALSE, verbose = FALSE)

cbind(
  known = coef(mod_known, which_param = 1)[[1]],
  unknown = coef(mod_unknown, which_param = 1)[[1]],
  manual = coef(mod_manual, which_param = 1)[[1]]
)


## --------------------------------------------------------------------------------------------------
airbnb$image <- paste0("/path/to/airbnb/airbnb/data/pictures/32/",
                       airbnb$id, ".jpg")


## --------------------------------------------------------------------------------------------------
cnn_block <- function(filters, kernel_size, pool_size, rate, input_shape = NULL){
    function(x){
      x %>% 
        layer_conv_2d(filters, kernel_size, padding="same", input_shape = input_shape) %>% 
        layer_activation(activation = "relu") %>% 
        layer_batch_normalization() %>% 
        layer_max_pooling_2d(pool_size = pool_size) %>% 
        layer_dropout(rate = rate)
    }
  }


## --------------------------------------------------------------------------------------------------
cnn <- cnn_block(filters = 16, kernel_size = c(3,3), pool_size = c(3,3), rate = 0.25,
                 shape(200, 200, 3))
deep_model_cnn <- function(x){
    x %>% 
    cnn() %>%
    layer_flatten() %>% 
    layer_dense(32) %>% 
    layer_activation(activation = "relu") %>% 
    layer_batch_normalization() %>% 
    layer_dropout(rate = 0.5) %>% 
    layer_dense(1)
}


## --------------------------------------------------------------------------------------------------
mod_cnn <- deepregression(
  y = y,
  list_of_formulas = list(
    ~1 + room_type + bedrooms + beds + 
      deep_model_cnn(image), 
    ~1 + room_type), 
  data = airbnb,
  list_of_deep_models = list(deep_model_cnn = list(deep_model_cnn, c(200,200,3))),
  optimizer = optimizer_adam(lr = 0.0001)
)


## ---- eval=FALSE-----------------------------------------------------------------------------------
## mod_cnn %>% fit(
##   epochs = 100,
##   early_stopping = TRUE,
##   patience = 5,
##   verbose = FALSE)

## --------------------------------------------------------------------------------------------------
coef(mod_cnn)

