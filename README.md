# Araberta Code Documentation
## Code Structure
This code consists of five main parts:
### 1-Setting up the Tokenizer:
Where a list of arabic letters and diactrictics is made and added to the tokenizer by hand. This method was chosen to fit our task since we want the model to predict diacritics only. 
Example:  we need the prediction to be a diacritic not some letters. ليك (mask) السلام ع
### 2-Roberta model:
A roberta model is used to train the data. Roberta needs three objects to be initialised:
- Roberta tokenizer: a specefic tokenizer class made by huggingface that extends the basic functionalities of a tokenizer, this class can be initialised from a pretrained tokenizer but in our case (adding the tokens by hand) it is still not functioning well but it will be fixed in the future patches.
- Configuration class: where some hyperparameters as vocab size, number of attention heads and number of hidden layers are defined. In earlier versions of transformers these parameters were defined in the config.yml file.
- Instance of the roberta model: this instance is initialized by one of two ways, either from a pretrained model where the model parameters are loaded from a training check point or from the configuration class mentioned above which creates a new model with randomized parameters.
> There are many variants of roberta model such as RobertaForQuestionAnswering, RobertaForSummarization and RobertaForCasualLM. For our task we use RobertaForMaskedLM. 
### 3-Preparing the dataset:
The dataset is divided into three portions, train, evaluation and test. The corpus used is from [Tashkeela](https://www.kaggle.com/hamzaabbad/tashkeela-processed-fully-diacritized-arabic-text) dataset which is composed of 75.6 million diacritized words, this data is not filtered or organised so we took an extra step of removing english and undiacritized rows from data. We also took the average length of the sentence into consideration when choosing the input layer size.Finally, we passed all the filtered data into the tokenizer which outputs, for each sentence, an input_IDs vector and an attention_mask vector.The sentence is either truncated or padded to a length of 512 tokens. This is the input required by the model. 
### 4-Data collator:
This object takes the tokenizer and the MLM probability as inputs. It's used to mask 15% of the input tokens in order to train the model, Where 15% gives the best accuracy. The data collator normally masks random tokens however, it doesn't mask special tokens as those marking the beginning and end of sentences. We used this property for our model where we prevented masking letters as we need the model to only predict diacritics.
### 5-Training the model:
The transformer's model is implemented using pytorch so it can be trained using:
1. AdamW optimizer.
2. Cross entropy loss criterion.
3. Learning rate scheduler with initial learning rate 5e-5 and no warm up steps.
The learning loop implementation can be found in the following link [pytorch tutorial] (https://pytorch.org/tutorials/beginner/pytorch_with_examples.html) 
Luckily this entire loop is implemented by Hugging face in the trainer class. We chose the training configuration and initialize an instance of this class
> Please use pytorch version 1.8.1 to avoid the implementation warnings in this class.


## Dependencies
- torch == 1.8.1
- tokenizer == 3.1.2
- tqdm == 4.56.0
- transformers == 4.6.1
- datasets == 1.7.0
- pandas == 1.2.4

## Classes And Function

### Tokenizer
`ByteLevelBPETokenizer()` : Creates a tokenizer class with 
```python
.add_token(tokens: List[Union[str, tokenizers.AddedToken]])
```
```python
.train(files: Union[str, List[str]],
    vocab_size: int = 30000,
    min_frequency: int = 2,
    show_progress: bool = True,
    special_tokens: List[Union[str, tokenizers.AddedToken]] = [],)
```
`RobertaTokenizerFast.from_pretrained()` : Creates a spicific tokenizer class for roberta tokenizer with
```python
pretrained_model_name_or_path: Union[str, os.PathLike],
```
### DataSets and Collators
`load_dataset()` : Loads a Dataset from a file
`DatasetDict()` : Creates a dictionary of datasets, please use this instead of a normal dictionary as it includes a lot more usable functionss
`DataCollatorForLanguageModeling()` : This is the main class we try to change, it fills a tensor the same shape of inputs with a prpability 0->1. Then applies a Bernolli function to the tensor. This produces a new boolean tensor with random True values at different indeces. Before it applies the bernolli function it reduces the probability of the indices of special_tokens to zero. We make use of this by reducing the probability of letters as well to zero.
`create_diacritization_variants()` : Creates a random number of variants of any string with randomly stripped diacritizations. Each new variant is more stripped down.

### The model
`RobertaForMaskedLM()` : Creates an instance of Roberta model that extends nn.module from pytorch, you can call the class using 
```python
RobertaForMaskedLM(
config (:class:`~transformers.RobertaConfig`): Model configuration class with all the parameters of the
    model. Initializing with a config file does not load the weights associated with the model, only the
    configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
    weights.
    )
```
or using 
```python
RobertaForMaskedLM.from_pretrained(pretrained_model_name_or_path: Union[str, os.PathLike, NoneType])
```
