# Sequence Models - Week 3 - Sequence-to-Sequence Models

## Basic Models

- e.g. Machine translation
  - encoder network for original language x 
    - produces encoding / embedding of original sentence
  - feed this encoded sentence into a decoder network to then make the translation
  - this actually works decently well
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_encoder_decoder.png)
- Image captioning
  - can use CNN for decoder
  - use RNN for encoder
  - works pretty well, esp. if captioning is not too long
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_image_captioning.png)
- but you don't want a randomly chosen caption or translation, you want the most likely one...

## Picking the most likely sentence

- decoder model is very similar to language model, except that instead of starting with an array of all zeros for x, it starts with a decoded array (green in img below)
- Probability of English sentence conditioned on an input French sentence
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_conditional_machine_translation.png)
- need to maximize likelihood of probabilities to get the most probable sentence
  - i.e. don't randomly sample from the probability distribution; otherwise, you'd end up with some okay but imperfect translations and possibly some bad translations
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_conditional_machine_translation2.png)
- Why not just use a greedy search?
  - i.e. choose most likely first word, then most likely second word, etc
  - i.e. maximizing joint probability
  - chance of "going" higher than chance of "visiting" (bc going is more common word)
  - but this ends up in less optimal translation than the first example in the img below
  - (argmax portion of this image added to describe what approximate searches try to do)
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_greedy_search.png)
- most common search method is an approximate search that finds argmax-ish of total sentence

## Beam Search

- say, you have 10K vocab
- beam width parameter
  - set it to, say, 3
  - adds 3 most likely words for each token
  - keep these 3 words in memory
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_beam_width.png)
- then, for translated_word_2, it will consider all 3/beam_width previous words from translated_word_1
  - find pair of tw1 and tw2 that is most likely given word 1 and french sentence
  - do this for all 3/beam_width previous words
  - 3 * 10k (i.e. vocab_size) comparisons made, and you choose the 3/beam_width best
  - this can cancel out possible choices for first word (e.g. image doesn't choose September as first word since it's not included in 3 best possibilities for tw2)
  - instantiate 3 copies of the network to search through 10K/vocab_size words at each copy (no need to make 30K copies of the network!)
  - x in the image is the entire French sentence (not just 1 word)
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_beam_search.png)
- terminates with EOS tag
- if Beam_width = 1, it will become greedy search discussed above
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_beam_search2.png)

## Refinements to Beam Search

- Length normalization
  - small change that can get much better results
  - numerical underflow an issue by raw multiplication of a lot of small numbers
  - in practice, we take logs
  - the log of a product becomes the sum of the logs
  --> numerically stable algorithm less prone to rounding errors
  - maximizing log should be the same as maximizing non-log, since log is monotonically increasing
  - without length normalization, probabilities of a short sentence is higher than longer sentence (log or no log) so w-out normalization the algorithm unnaturally prefers shorter sentences
  --> normalize probability by length of sentence
  - divide by t_y<sup>alpha</sup> where alpha is a parameter
  - alpha is a hack/hieuristic that works well in practice... try a few in practice and see what works best
  - you see a lot of sentences with length 1,2,3 and run it all the way up to 30 length
  - look at all output sentences and score them with cost function on the bottom of the img
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_length_normalization.png)
- choosing beam width length
  - performance vs computational cost trade off
  - common to see beam width of 10 in production
  - 100 high, but seen in some production systems
  - 1000 for some some research applications
  - diminishing returns: huge gains going 1-10, less so higher up
  - beam search isn't guaranteed to find exact maximum, but runs much faster than bread first search or depth first search
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_beam_width_param_choosing.png)
- error analysis is one of most important ways of improving algorithm (some simple things can help)

## Error analysis in beam search

- how do determine if beam search or RNN is causing errors?
- always tempting to increase beam width
- compare probabilities computed by RNN to probabilities computed by beam search to determine which was bigger in order to determine next steps / ascribe particular error to RNN or beam search
- y<sup>*</sup> is human translation, y<sup>^</sup> is beam search y
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_beamW_error_analysis.png)
- case 1: if prob human translation is higher than beam search prob, beam search is at fault
- case 2: if beam search prob is gte human translation prob, RNN is at fault
- need to take into account length normalization to do this properly
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_beamW_error_analysis2.png)
- go through dev set and find the mistakes, determining which RNN or BS is at fault
- carry out error analysis to figure out fraction of errors due to RNN (or BS)
- only if you find BS is responsible for most of the errors do you increase beam width
- in contrast, if RNN is at fault, do a deeper error analysis related to RNN (see Week 3)
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_beamW_error_analysis3.png)

## Bleu Score (optional)

- multiple equally good answers in machine translation
- how do you evaluate machine translation system if there are multiple acceptable/equally good answers
- Bleu score (Bilingual evaluation understudy)
  - automatically computes a score of a translation
  - as long as translation is close to a human generated reference (as part of dev/test sets)
  - "understudy" for having humans review every translation
  - precision: if machine translation in human refs, what fraction of words that you predicted are in refs
  - modified precision: "the" gets credit up to the max number of times "the" is in one of the human generated refs
- bigrams portion of blue score:
  - compute modified precision for bigrams
  - sum of matching bigrams / sum of bigrams in machine translation
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_bleu_score.png)
- perfect score is exactly matching translation
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_bleu_score2.png)
- BP = brevity penalty
  - penalizes if machine translation length is less than human translation length
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_bleu_score3.png)
- useful single evaluation metric --> moved progress forward
- not used in speech recognition (bc only 1 right answer)
- used in machine translation and image captioning

## Attention Model Intuition

- works better than encoder / decoder
- very important idea in ML
- encoder / decoder "memorizes" entire sentence and then translates/decodes it from scratch
- encoder / decoder works well for short sentences but is bad for longer sentences (hard for it to "memorize" it all)
- attention model is green line of performance below
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_attention_issue.png)
- attention model is very influential paper - Bahdanau et al 2014 "neural machine translation by jointly learning ..."
- BRNN
- attention model computes attention weights
- in first word, you don't need to be looking way at the end of the sentence
- alpha<sup>(1,1)</sup>, alpha<sup>(1,2)</sup>....alpha<sup>(1,len_text)</sup>
- for first word, this combination of weights determines the context of the first word
- new alpha array / context for second word and so on
- activation depends on context of current row and state (s) of previous step
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_attention_intuition.png)

## Attention Model

- attention model allows NN to pay attention to only part of the sentence much like how a human translator would
- a<sup\<6\></sup> for backprop is initialize as a vector of all zeros
- a<sup\<t'\></sup> is the concatenated of forward occurence activations and backward occurence activations for BRNN of French sentence (original sentence)
- alpha is the attention parameter, tells us how much the context will depend on the attention weights
- C is weighted sum of attention and activations
- alpha<sup>\<t, t'\></sup> is the amount of attention y<sup>\<t\></sup> should pay to a<sup>\<t'\></sup>
  - when generating output y_t, how much attention should you be paying to the t' input word? (alpha tells you this)
- alphas change for every input word
- state and context (top left) look like a pretty 'normal' RNN predicting one word at a time
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_attention_model.png)
- for every fixed value of t, alphas sum to 1 (see denominator in a_tt' equation
- how to compute exponential values
  - one way is to use a small NN
  - only 1 hidden layer bc you calculate these a lot
  - leave details to backprop
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_computing_alphas.png)
- this actually works, learns to pay attention to the right things
- takes quadratic cost to run this algorithm
  - but in machine translation, most texts are pretty small
- also applied to image captioning
  - pays attention to parts of the picture at a time
- date normalization
- visualization of attention weights/alphas
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_attention_examples.png)

## Speech recognition

- audio clip to transcript
- preprocessing - create a spectrogram from audio clip
  - how loud is this sound at different frequencies at different times
- no need for phonemes
  - in academia, 3000H of audio would be reasonable
  - in production systems, 100,000H of audio makes deep learning work better
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_speech_recognition_problem.png)
- CtC cost for speech recognition works well
  - Connectionist temporal classification
  - in practice, will be Bi-GRU/LStM
  - # input time steps is much larger than output
  - collapse repeated characters not separated by blank
  - skip blanks/underscores
  - NN has 1000 outputs in this example, but CtC reduces effective output to reasonable characters/words
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_ctc_cost.png)

## Trigger Word Detection

- put a few ones after when trigger word was said
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/5_SequenceModels/week3/5wk3_trigger_word_detection.png)


