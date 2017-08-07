########################################

p_grid <- seq(from=0, to=1, length.out=1000)
prior <-  rep(1,1000)
likelihood <- dbinom(6, size=9, prob=p_grid)
unstd.posterior <- likelihood * prior 
posterior <- unstd.posterior / sum(unstd.posterior)

samples <- sample( p_grid, prob=posterior, size=1e4, replace=TRUE)
plot(samples)
dens(samples)
sum(samples < 0.5) / 1e4
sum(samples >0.5 & samples < 0.75) / 1e4

########################################

quantile(samples, 0.8)
quantile(samples, c(0.1,0.9))

########################################

p_grid <- seq(from=0, to=1, length.out=1000)
prior <-  rep(1,1000)
likelihood <- dbinom(3, size=3, prob=p_grid)
unstd.posterior <- likelihood * prior 
posterior <- unstd.posterior / sum(unstd.posterior)
samples <- sample( p_grid, prob=posterior, size=1e4, replace=TRUE)
dens(samples)
PI(samples, prob=0.5)
HPDI(samples,prob=0.5)

########################################

p_grid[ which.max(posterior)] # mode
chainmode(samples, adj=0.01) # mode
mean(samples)
median(samples)

########################################

sum( posterior*abs(0.5-p_grid))
loss <-sapply( p_grid, function(d) sum( posterior*abs(d-p_grid)))
plot(loss)
p_grid[which.min(loss)] # same as median

########################################

dbinom(0:2, size=2, prob=0.7)

rbinom(10, size=2, prob=0.7)

dummy_w <- rbinom(1e5, size=2, prob=0.7)
table(dummy_w)/1e5

dummy_w <- rbinom(1e5, size=9, prob=0.7)
simplehist(dummy_w, xlab="dummy water count")

########################################

w <- rbinom(1e4, size=9, prob=0.6)
w <- rbinom(1e4, size=9, prob=samples)
dens(samples)
simplehist(w, ylab="posterior predictive distribution")
