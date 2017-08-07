ways <- c(0, 3, 8, 9, 0)
ways/sum(ways)

########################################
dbinom(6, size=9, prob = 0.5)

########################################
p_grid <- seq(from=0, to=1, length.out=20)
prior <-  exp(-5*abs(p_grid-0.5))# ifelse( p_grid < 0.5, 0, 1) #rep(1,20)
likelihood <- dbinom(6, size=9, prob=p_grid)
unstd.posterior <- likelihood * prior 
posterior <- unstd.posterior / sum(unstd.posterior)
plot(p_grid, posterior, type="b",
     xlab="probability of water", ylab="posterior probability")
mtext("20 points")

########################################
library(rethinking)
globe.qa <- map(
  alist(
    w ~ dbinom(9,p), # likelihood
    p ~ dunif(0,1)   # prior
  ),
  data=list(w=6)
)
precis(globe.qa)

########################################

w <- 6
n <- 9
curve(dbeta(x, w+1, n-w+1), from=0, to=1)
curve(dnorm(x,0.67,0.16), lty=2, add=TRUE)
