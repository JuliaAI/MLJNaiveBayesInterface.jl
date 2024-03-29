using MLJNaiveBayesInterface
using MLJBase
using Test

import StableRNGs
rng = StableRNGs.StableRNG(1234)


## GAUSSIAN

gaussian_classifier = GaussianNBClassifier()

# gaussian classifier takes continuous features
X, y = @load_iris

train, test = partition(eachindex(y), 0.6)

fitresultG, cacheG, reportG = fit(gaussian_classifier, 1,
                                  selectrows(X, train), y[train]);

# test matrix is okay as input:
fitresultGM, _ = fit(gaussian_classifier, 1,
                     selectrows(MLJBase.matrix(X), train), y[train]);
@test fitresultGM.gaussians == fitresultG.gaussians
@test all([:c_counts, :gaussians, :n_obs]) do field
    getproperty(fitresultG, field) ==  getproperty(fitresultGM, field)
end

params = fitted_params(gaussian_classifier, fitresultG)
@test params.n_obs == length(train)

gaussian_pred = predict(gaussian_classifier,
                        fitresultG,
                        selectrows(X, test));

# test matrix is okay as input:
gaussian_predM = predict(gaussian_classifier,
                        fitresultG,
                        selectrows(MLJBase.matrix(X), test));
@test pdf(gaussian_predM, levels(y)) == pdf(gaussian_pred, levels(y))

yhat1 = gaussian_pred[1]
@test Set(classes(yhat1)) == Set(classes(y[1]))

# test with linear data:
x1 = randn(rng, 3000);
x2 = randn(rng, 3000);
x3 = randn(rng, 3000);
X = (x1=x1, x2=x2, x3=x3);
ycont = x1 - x2 -2x3;
ydiscrete = map(ycont) do η
    η > 0 ? "go" : "stop"
end
y = coerce(ydiscrete, Multiclass)
train, test = partition(eachindex(y), 0.8);

gaussian_classifier = GaussianNBClassifier()

fitresultG, cacheG, reportG = MLJBase.fit(gaussian_classifier, 1,
             selectrows(X, train), y[train])

params = fitted_params(gaussian_classifier, fitresultG)
@test params.n_obs == length(train)

gaussian_pred = MLJBase.predict_mode(gaussian_classifier,
                                     fitresultG, selectrows(X, test))

@test sum(gaussian_pred .!= y[test])/length(y) < 0.05


## MULTINOMIAL

# first contrive some test data

# some word counts in children's books about colours:
red = [2, 0, 1, 0, 1]
blue = [4, 1, 2, 3, 2]
green = [0, 2, 0, 6, 1]
X = (red=red, blue=blue, green=green)

# gender of author:
y = categorical(['m', 'f', 'm', 'f', 'm'])
male = y[1]
female = y[2]

# Note: The smoothing algorithm is to add to the training data, for
# each class observed, a row with every feature getting count of
# alpha. So smoothing also effects the class marginals (is this
# standard?) Only integer values of alpha allowed.

# computing conditional probabilities by hand with Lagrangian
# smoothing (alpha=1):
red_given_m = 5/16
blue_given_m = 9/16
green_given_m = 2/16
red_given_f = 1/15
blue_given_f = 5/15
green_given_f = 9/15

m_(red, blue, green) =
    4/7*(red_given_m^red)*(blue_given_m^blue)*(green_given_m^green)
f_(red, blue, green) =
    3/7*(red_given_f^red)*(blue_given_f^blue)*(green_given_f^green)
normalizer(red, blue, green) = m_(red, blue, green) + f_(red, blue, green)
m(a...) = m_(a...)/normalizer(a...)
f(a...) = f_(a...)/normalizer(a...)

Xnew = (red=[1, 1], blue=[1, 2], green=[1, 3])

# prediction by hand:

yhand =[MLJBase.UnivariateFinite([male, female], [m(1, 1, 1), f(1, 1, 1)]),
        MLJBase.UnivariateFinite([male, female], [m(1, 2, 3), f(1, 2, 3)])]

multinomial_classifier = MultinomialNBClassifier()

fitresultMLT, cacheMLT, reportMLT =
    MLJBase.fit(multinomial_classifier, 1, X, y)

# test with matrix input:
fitresultMLTM, _ =
    MLJBase.fit(multinomial_classifier, 1, MLJBase.matrix(X), y)
@test all([:c_counts, :n_obs, :x_counts, :x_totals]) do field
    getproperty(fitresultMLTM, field) ==  getproperty(fitresultMLT, field)
end

params = fitted_params(multinomial_classifier, fitresultMLT)
@test params.n_obs == length(y) + 3 * 2 # 3 features, 2 classes

yhat = MLJBase.predict(multinomial_classifier, fitresultMLT, Xnew)

# test with matrix input:
yhatM = MLJBase.predict(multinomial_classifier, fitresultMLT, MLJBase.matrix(Xnew))
@test pdf(yhat, levels(y)) == pdf(yhatM, levels(y))

# see issue https://github.com/dfdx/NaiveBayes.jl/issues/42
@test pdf(yhand[1], 'm') ≈ pdf(yhat[1], 'm')
@test pdf(yhand[1], 'f') ≈ pdf(yhat[1], 'f')
