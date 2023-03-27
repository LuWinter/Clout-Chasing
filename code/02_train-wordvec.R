
#########################################################
# SCRIPT: 02_train-wordvec.R
# TASK: This script trains the wordvec using "Glove" algorithm
#########################################################


# PACKAGES ----------------------------------------------------------------
pacman::p_load(text2vec)
pacman::p_load(jiebaR)
pacman::p_load(arrow)
pacman::p_load(dplyr)
pacman::p_load(stringr)


# 1. Split Tokens ---------------------------------------------------------
## Split tokens using jiebaR
word_segment <- worker(stop_word = "data/baidu_stopwords.txt")

## Conduct splitting by year
res_list = list()
for (year in 2016:2022) {
  print(paste("Now process year", year))
  
  print("=> Collect text...")
  path = paste0("data/HuDongE/Year=", year)
  hde_arrow <- open_dataset(sources = path)
  hde_text <- hde_arrow |> 
    select(提问内容, 回复内容) |> 
    collect()
  
  print("=> Combine text...")
  size <- nrow(hde_text)
  hde_text <- str_flatten(
    string = Reduce(paste, hde_text[1:size, ])
  )
  
  print("=> Split tokens...")
  tokens <- segment(
    code = str_remove_all(hde_text, pattern = "[0-9]"),
    jiebar = word_segment
  )
  
  res_list <- append(res_list, unlist(tokens))
  rm(path, hde_arrow, hde_text, size, tokens)
}
res_list <- list(unlist(res_list))


# 2. Train Wordvec --------------------------------------------------------
## Build corpus
it = itoken(res_list, progressbar = TRUE)
vocab = create_vocabulary(it)
vocab = prune_vocabulary(vocab, term_count_min = 4L)

## Calculate co-occurrence matrix
vectorizer = vocab_vectorizer(vocab)
tcm = create_tcm(it, vectorizer, skip_grams_window = 5L)

saveRDS(object = res_list, file = "processed/tokens16-22.rds")
rm(res_list)

## Generate wordvec
glove = GlobalVectors$new(rank = 50, x_max = 10)
wv_main = glove$fit_transform(tcm, n_iter = 10, convergence_tol = 0.01, n_threads = 8)

wv_context = glove$components
word_vectors = wv_main + t(wv_context)

## Test wordvec 
find_similar_words <- function(keyword) {
  target_word = word_vectors[keyword, , drop = FALSE]
  cos_sim = sim2(
    x = word_vectors, 
    y = target_word, 
    method = "cosine", 
    norm = "l2"
  )
  head(sort(cos_sim[, 1], decreasing = TRUE), 5)
}
find_similar_words(keyword = "一年")

## Save wordvec as rds file
saveRDS(object = tcm, file = "processed/tcm16-22.rds")
saveRDS(object = word_vectors, file = "processed/word_vectors16-22.rds")


### EOF
