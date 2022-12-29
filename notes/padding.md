


# https://towardsdatascience.com/padding-sequences-with-simple-min-max-mean-document-encodings-aa27e1b1c781

```
# Encode the documents using the new embedding
encoded_docs = [[w2v_model.wv[word] for word in post] for post in documents]
```


```
padded_posts = []

for post in encoded_docs:
    # Pad short posts with alternating min/max
    if len(post) < MAX_LENGTH:

        # Method 1
        pointwise_min = np.minimum.reduce(post)
        pointwise_max = np.maximum.reduce(post)
        padding = [pointwise_max, pointwise_min]

        # Method 2
        pointwise_avg = np.mean(post)
        padding = [pointwise_avg]

        post += padding * ceil((MAX_LENGTH - len(post) / 2.0))

    # Shorten long posts or those odd number length posts we padded to 51
    if len(post) > MAX_LENGTH:
        post = post[:MAX_LENGTH]

    # Add the post to our new list of padded posts
    padded_posts.append(post)
```