module MLJNaiveBayesInterface

export GaussianNBClassifier, MultinomialNBClassifier, HybridNBClassifier

import LogExpFunctions
import MLJModelInterface
import NaiveBayes
import MLJModelInterface: Table, Continuous, Count, Finite, OrderedFactor,
    Multiclass

const MMI = MLJModelInterface
const PKG = "MLJNaiveBayesInterface"


## GAUSSIAN NAIVE BAYES CLASSIFIER

mutable struct GaussianNBClassifier <: MMI.Probabilistic
end

function MMI.fit(model::GaussianNBClassifier, verbosity::Int
                 , X
                 , y)
    Xmatrix = MMI.matrix(X) |> permutedims
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
    Xmatrix = MMI.matrix(Xnew) |> permutedims

    classes_observed, logprobs = NaiveBayes.predict_logprobs(fitresult, convert(Matrix{Float64}, Xmatrix))
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

    Xmatrix = MMI.matrix(X) |> permutedims
    p = size(Xmatrix, 1)

    yplain = convert(Vector, y) # ordinary Vector
    classes_observed = unique(yplain)

    res = NaiveBayes.MultinomialNB(classes_observed, p ,alpha= model.alpha)
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
    Xmatrix = MMI.matrix(Xnew) |> permutedims

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
MMI.package_name(::Type{<:GaussianNBClassifier}) = "NaiveBayes"
MMI.package_uuid(::Type{<:GaussianNBClassifier}) =
    "9bbee03b-0db5-5f46-924f-b5c9c21b8c60"
MMI.package_url(::Type{<:GaussianNBClassifier}) =
    "https://github.com/dfdx/NaiveBayes.jl"
MMI.is_pure_julia(::Type{<:GaussianNBClassifier}) = true
MMI.input_scitype(::Type{<:GaussianNBClassifier}) = Table(Continuous)
MMI.target_scitype(::Type{<:GaussianNBClassifier}) = AbstractVector{<:Finite}

MMI.load_path(::Type{<:MultinomialNBClassifier}) =
    "$PKG.MultinomialNBClassifier"
MMI.package_name(::Type{<:MultinomialNBClassifier}) = "NaiveBayes"
MMI.package_uuid(::Type{<:MultinomialNBClassifier}) =
    "9bbee03b-0db5-5f46-924f-b5c9c21b8c60"
MMI.package_url(::Type{<:MultinomialNBClassifier}) =
    "https://github.com/dfdx/NaiveBayes.jl"
MMI.is_pure_julia(::Type{<:MultinomialNBClassifier}) = true
MMI.input_scitype(::Type{<:MultinomialNBClassifier}) = Table(Count)
MMI.target_scitype(::Type{<:MultinomialNBClassifier}) = AbstractVector{<:Finite}

"""
$(MMI.doc_header(GaussianNBClassifier))


`GaussianNBClassifier`: Gaussian Naive Bayes classifier. Assumes variables have a
multivariate normal distribution. Good for real-valued data.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

Where

- `X`: is any table of input features (eg, a `DataFrame`) whose columns
  are of scitype `Continuous`; check the scitype with `schema(X)`

- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `Finite`; check the scitype with `schema(y)`

Train the machine using `fit!(mach, rows=...)`.

# Operations

- `predict(mach, Xnew)`: return predictions of the target given new
  features `Xnew` having the same Scitype as `X` above. Predictions are probabilistic but
  uncalibrated.
- `predict_mode(mach, Xnew)`: Return the mode of above predictions.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `c_counts`: A dictionary containing the observed count of each input class.
- `c_stats`: A dictionary containing observed statistics on each input class. Each class is
  represented by a `DataStats` object, with the following fields:
    - `n_vars`: The number of variables used to describe the class's behavior.
    - `n_obs`: The number of times the class is observed.
    - `obs_axis`: The axis along which the observations were computed.
- `gaussians`: A per class dictionary of gaussians, each representing the distribution of
  the class.
- `n_obs`: The total number of seen observations.

# Examples

```
using MLJ
GaussianNB = @load GaussianNBClassifier pkg=NaiveBayes

X, y = @load_iris
clf = GaussianNB(k=3)
mach = machine(clf, X, y) |> fit!

fitted_params(mach)

preds = predict(mach, X);
preds[1]
predict_mode(mach, X)
```

See also
[`MultinomialNBClassifier`](@ref)
"""
GaussianNBClassifier
"""
$(MMI.doc_header(MultinomialNBClassifier))


`MultinomialNBClassifier`: MultinomialNBClassifier Naive Bayes classifier.
Assumes variables have a multinomial distribution. Good for counts and text classification.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

Where

- `X`: is any table of input features (eg, a `DataFrame`) whose columns
  are of scitype `Count`; check the scitype with `schema(X)`

- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `Finite`; check the scitype with `schema(y)`

Train the machine using `fit!(mach, rows=...)`.

# Operations

- `predict(mach, Xnew)`: return predictions of the target given new
  features `Xnew` having the same Scitype as `X` above. Predictions are probabilistic but
  uncalibrated.
- `predict_mode(mach, Xnew)`: Return the mode of above predictions.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `c_counts`: A dictionary containing the observed count of each input class.
- `c_stats`: A dictionary containing observed statistics on each input class. Each class is
  represented by a `DataStats` object, with the following fields:
    - `n_vars`: The number of variables used to describe the class's behavior.
    - `n_obs`: The number of times the class is observed.
    - `obs_axis`: The axis along which the observations were computed.
- `gaussians`: A per class dictionary of gaussians, each representing the distribution of
  the class.
- `n_obs`: The total number of seen observations.

# Examples

```
using MLJ
GaussianNB = @load GaussianNBClassifier pkg=NaiveBayes
```

See also
[`GaussianNBClassifier`](@ref)
"""
MultinomialNBClassifier

end # module
