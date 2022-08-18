module MLJNaiveBayesInterface

export GaussianNBClassifier, MultinomialNBClassifier, HybridNBClassifier

import LogExpFunctions
import MLJModelInterface
import NaiveBayes
import MLJModelInterface: Table, Continuous, Count, Finite, OrderedFactor,
    Multiclass

const MMI = MLJModelInterface
const PKG = "MLJNaiveBayesInterface"


matrix(X) = MMI.matrix(X)
matrix(X::AbstractMatrix) = X

## GAUSSIAN NAIVE BAYES CLASSIFIER

mutable struct GaussianNBClassifier <: MMI.Probabilistic
end

function MMI.fit(model::GaussianNBClassifier, verbosity::Int
                , X
                , y)
    Xmatrix = matrix(X)'
    p = size(Xmatrix, 1)

    yplain = convert(Vector, y) # y as Vector
    classes_seen = unique(yplain)

    # initiates dictionaries keyed on classes_seen:
    res = NaiveBayes.GaussianNB(classes_seen, p)

    fitresult = NaiveBayes.fit(res, convert(Matrix{Float64}, Xmatrix), yplain)

    report = NamedTuple{}()

    return fitresult, nothing, report

end

function MMI.fitted_params(::GaussianNBClassifier, fitresult)
    return (c_counts=fitresult.c_counts,
            c_stats=fitresult.c_stats,
            gaussians=fitresult.gaussians,
            n_obs=fitresult.n_obs)
end

function MMI.predict(model::GaussianNBClassifier, fitresult, Xnew)
    Xmatrix = matrix(Xnew)'

    classes_observed, logprobs = NaiveBayes.predict_logprobs(
        fitresult,
        convert(Matrix{Float64}, Xmatrix)
    )
    # Note that NaiveBayes does not normalize the probabilities.

    # Normalize probabilities
    for p in (view(logprobs, :, i) for i in axes(logprobs, 2))
        LogExpFunctions.softmax!(p, p)
    end
    probs = logprobs

    # UnivariateFinite constructor automatically adds unobserved
    # classes with zero probability. Note we need to use adjoint here:
    return MMI.UnivariateFinite(collect(classes_observed), probs')
end


## MULTINOMIAL NAIVE BAYES CLASSIFIER

mutable struct MultinomialNBClassifier <: MMI.Probabilistic
    alpha::Int
end

function MultinomialNBClassifier(; alpha=1)
    m = MultinomialNBClassifier(alpha)
    return m
end

function MMI.fit(model::MultinomialNBClassifier, verbosity::Int
                 , X
                 , y)

    Xmatrix = matrix(X) |> permutedims
    p = size(Xmatrix, 1)

    yplain = convert(Vector, y) # ordinary Vector
    classes_observed = unique(yplain)

    res = NaiveBayes.MultinomialNB(classes_observed, p, alpha= model.alpha)
    fitresult = NaiveBayes.fit(res, convert(Matrix{Int}, Xmatrix), yplain)

    report = NamedTuple()

    return fitresult, nothing, report
end

function MMI.fitted_params(::MultinomialNBClassifier, fitresult)
    return (c_counts=fitresult.c_counts,
            x_counts=fitresult.x_counts,
            x_totals=fitresult.x_totals,
            n_obs=fitresult.n_obs)
end

function MMI.predict(model::MultinomialNBClassifier, fitresult, Xnew)
    Xmatrix = matrix(Xnew) |> permutedims

    # Note that NaiveBayes.predict_logprobs returns probabilities that
    # are not normalized.

    classes_observed, logprobs =
        NaiveBayes.predict_logprobs(fitresult, convert(Matrix{Int}, Xmatrix))

    # Normalize probabilities
    for p in (view(logprobs, :, i) for i in axes(logprobs, 2))
        LogExpFunctions.softmax!(p, p)
    end
    probs = logprobs

    return MMI.UnivariateFinite(collect(classes_observed), probs')
end


## METADATA

MMI.load_path(::Type{<:GaussianNBClassifier}) =
    "$PKG.GaussianNBClassifier"
MMI.human_name(::Type{<:GaussianNBClassifier}) = "Gaussian naive Bayes classifier"
MMI.package_name(::Type{<:GaussianNBClassifier}) = "NaiveBayes"
MMI.package_uuid(::Type{<:GaussianNBClassifier}) =
    "9bbee03b-0db5-5f46-924f-b5c9c21b8c60"
MMI.package_url(::Type{<:GaussianNBClassifier}) =
    "https://github.com/dfdx/NaiveBayes.jl"
MMI.is_pure_julia(::Type{<:GaussianNBClassifier}) = true
MMI.input_scitype(::Type{<:GaussianNBClassifier}) =
    Union{AbstractMatrix{<:Continuous}, Table(Continuous)}
MMI.target_scitype(::Type{<:GaussianNBClassifier}) = AbstractVector{<:Finite}

MMI.load_path(::Type{<:MultinomialNBClassifier}) =
    "$PKG.MultinomialNBClassifier"
MMI.human_name(::Type{<:MultinomialNBClassifier}) = "multinomial naive Bayes classifier"
MMI.package_name(::Type{<:MultinomialNBClassifier}) = "NaiveBayes"
MMI.package_uuid(::Type{<:MultinomialNBClassifier}) =
    "9bbee03b-0db5-5f46-924f-b5c9c21b8c60"
MMI.package_url(::Type{<:MultinomialNBClassifier}) =
    "https://github.com/dfdx/NaiveBayes.jl"
MMI.is_pure_julia(::Type{<:MultinomialNBClassifier}) = true
MMI.input_scitype(::Type{<:MultinomialNBClassifier}) =
        Union{AbstractMatrix{<:Count}, Table(Count)}
MMI.target_scitype(::Type{<:MultinomialNBClassifier}) = AbstractVector{<:Finite}

"""
$(MMI.doc_header(GaussianNBClassifier))

Given each class taken on by the target variable `y`, it is supposed that the conditional
probability distribution for the input variables `X` is a multivariate Gaussian. The mean
and covariance of these Gaussian distributions are estimated using maximum likelihood, and a
probability distribution for `y` given `X` is deduced by applying Bayes' rule. The required
marginal for `y` is estimated using class frequency in the training data.

**Important.** The name "naive Bayes classifier" is perhaps misleading. Since we are
learning the full multivariate Gaussian distributions for `X` given `y`, we are not
applying the usual naive Bayes independence condition, which would amount to forcing the
covariance matrix to be diagonal.


# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns
  are of scitype `Continuous`; check the column scitypes with `schema(X)`

- `y` is the target, which can be any `AbstractVector` whose element
  scitype is `Finite`; check the scitype with `schema(y)`

Train the machine using `fit!(mach, rows=...)`.

# Operations

- `predict(mach, Xnew)`: return predictions of the target given new features `Xnew`, which
  should have the same scitype as `X` above. Predictions are probabilistic.

- `predict_mode(mach, Xnew)`: Return the mode of above predictions.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `c_counts`: A dictionary containing the observed count of each input class.

- `c_stats`: A dictionary containing observed statistics on each input class. Each class is
  represented by a `DataStats` object, with the following fields:

    - `n_vars`: The number of variables used to describe the class's behavior.

    - `n_obs`: The number of times the class is observed.

    - `obs_axis`: The axis along which the observations were computed.

- `gaussians`: A per class dictionary of Gaussians, each representing the distribution of
  the class. Represented with type `Distributions.MvNormal` from the Distributions.jl
  package.

- `n_obs`: The total number of observations in the training data.

# Examples

```
using MLJ
GaussianNB = @load GaussianNBClassifier pkg=NaiveBayes

X, y = @load_iris
clf = GaussianNB()
mach = machine(clf, X, y) |> fit!

fitted_params(mach)

preds = predict(mach, X) # probabilistic predictions
preds[1]
predict_mode(mach, X) # point predictions
```

See also
[`MultinomialNBClassifier`](@ref)
"""
GaussianNBClassifier

"""
$(MMI.doc_header(MultinomialNBClassifier))

The [multinomial naive Bayes
classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Multinomial_naive_Bayes) is
often applied when input features consist of a counts (scitype `Count`) and when
observations for a fixed target class are generated from a multinomial distribution with
fixed probability vector, but whose sample length varies from observation to
observation. For example, features might represent word counts in text documents being
classified by sentiment.


# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns are of scitype
  `Count`; check the column scitypes with `schema(X)`.

- `y` is the target, which can be any `AbstractVector` whose element scitype is `Finite`;
  check the scitype with `schema(y)`.

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters

- `alpha=1`: Lindstone smoothing in estimation of multinomial probability vectors from
  training histograms (default corresponds to Laplacian smoothing).


# Operations

- `predict(mach, Xnew)`: return predictions of the target given new features `Xnew`, which
  should have the same scitype as `X` above.

- `predict_mode(mach, Xnew)`: Return the mode of above predictions.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `c_counts`: A dictionary containing the observed count of each input class.

- `x_counts`: A dictionary containing the categorical counts of each input class.

- `x_totals`: The sum of each count (input feature), ungrouped.

- `n_obs`: The total number of observations in the training data.

# Examples

```
using MLJ
import TextAnalysis

CountTransformer = @load CountTransformer pkg=MLJText
MultinomialNBClassifier = @load MultinomialNBClassifier pkg=NaiveBayes

tokenized_docs = TextAnalysis.tokenize.([
    "I am very mad. You never listen.",
    "You seem to be having trouble? Can I help you?",
    "Our boss is mad at me. I hope he dies.",
    "His boss wants to help me. She is nice.",
    "Thank you for your help. It is nice working with you.",
    "Never do that again! I am so mad. ",
])

sentiment = [
    "negative",
    "positive",
    "negative",
    "positive",
    "positive",
    "negative",
]

mach1 = machine(CountTransformer(), tokenized_docs) |> fit!

# matrix of counts:
X = transform(mach1, tokenized_docs)

# to ensure scitype(y) <: AbstractVector{<:OrderedFactor}:
y = coerce(sentiment, OrderedFactor)

classifier = MultinomialNBClassifier()
mach2 = machine(classifier, X, y)
fit!(mach2, rows=1:4)

# probabilistic predictions:
y_prob = predict(mach2, rows=5:6) # distributions
pdf.(y_prob, "positive") # probabilities for "positive"
log_loss(y_prob, y[5:6])

# point predictions:
yhat = mode.(y_prob) # or `predict_mode(mach2, rows=5:6)`
```

See also
[`GaussianNBClassifier`](@ref)
"""
MultinomialNBClassifier

end # module
