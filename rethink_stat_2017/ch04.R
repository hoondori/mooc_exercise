
########################################
step = 4
pos <- replicate(1000, sum( runif(step,-1,1)))
dens(pos,norm.comp = TRUE)

growth <- replicate(10000, prod(1+runif(12,0,0.1)))
dens(growth, norm.comp = TRUE)

big <- replicate(10000, prod(1+runif(12,0,0.5)))
dens(big, norm.comp = TRUE) # somewhat different from normal

log.big <- replicate(10000, log(prod(1+runif(12,0,0.5))))
dens(log.big, norm.comp = TRUE) # somewhat different from normal

########################################

w <- 6; n <- 9;
p_grid <- seq(from=0, to=1, length.out=100)
likelihood <- dbinom(3, size=3, prob=p_grid)
posterior <- dbinom(w,n,p_grid) * dunif(p_grid,0,1)
posterior <- posterior / sum(posterior)
plot(posterior)

########################################

data(Howell1)
d <- Howell1 
str(d)
d$height
d2 <- d[ d$age >= 18, ]
curve(dnorm(x,178,20),from=100,to=250) # plot prior
curve(dunif(x,0,50),from=-10,to=60) # plot prior

sample_mu <- rnorm( 1e4, 178, 20 )
sample_sigma <- runif( 1e4, 0, 50 )
prior_h <- rnorm( 1e4, sample_mu, sample_sigma )
dens( prior_h )

########################################

mu.list <- seq( from=140, to=160, length.out=200)
sigma.list <- seq( from=4, to=9, length.out=200)
post <- expand.grid( mu=mu.list, sigma=sigma.list)
post$LL <- sapply( 1:nrow(post), 
                   function(i) sum( dnorm(d2$height, mean=post$mu[i], sd=post$sigma[i], log=TRUE) )
)
post$prod <- post$LL + dnorm( post$mu, 178, 20, TRUE) + dunif( post$sigma, 0, 50, TRUE)
post$prob <- exp( post$prod - max(post$prod))
          
contour_xyz( post$mu, post$sigma, post$prob )
image_xyz( post$mu, post$sigma, post$prob )


########################################

sample.rows <- sample( 1:nrow(post), size=1e4, replace=TRUE, prob=post$prob)
sample.mu <- post$mu[sample.rows]
sample.sigma <- post$sigma[ sample.rows ]
plot( sample.mu, sample.sigma, cex=0.6, pch=16, col=col.alpha(rangi2,0.1))
dens(sample.mu, norm.comp = TRUE)
dens(sample.sigma, , norm.comp = TRUE)
HPDI(sample.mu)
HPDI(sample.sigma)


########################################

d3 <- sample( d2$height, size=20)
mu.list <- seq( from=140, to=160, length.out=200)
sigma.list <- seq( from=4, to=9, length.out=200)
post2 <- expand.grid( mu=mu.list, sigma=sigma.list)
post2$LL <- sapply( 1:nrow(post2), 
                   function(i) sum( dnorm(d3, mean=post2$mu[i], sd=post2$sigma[i], log=TRUE) )
)
post2$prod <- post2$LL + dnorm( post2$mu, 178, 20, TRUE) + dunif( post2$sigma, 0, 50, TRUE)
post2$prob <- exp( post2$prod - max(post2$prod))
sample2.rows <- sample( 1:nrow(post2), size=1e4, replace=TRUE, prob=post2$prob)
sample2.mu <- post2$mu[sample2.rows]
sample2.sigma <- post2$sigma[ sample2.rows ]
plot( sample2.mu, sample2.sigma, cex=0.6, pch=16, col=col.alpha(rangi2,0.1))
dens(sample2.mu, norm.comp = TRUE)
dens(sample2.sigma, norm.comp = TRUE)

########################################

flist <- alist(
  height ~ dnorm( mu, sigma ),
  mu ~ dnorm( 178, 20 ),
  sigma ~ dunif( 0, 50 )
)
m4.1 <- map( flist, data=d2 )
precis(m4.1)
vcov(m4.1) # variance-covariance matrix
sqrt(0.084)
diag(vcov(m4.1)) # variance only
cov2cor(vcov(m4.1)) # correlation matrix


########################################

post <- extract.samples(m4.1, n=1e4)
head(post)
plot(post)

########################################

# use of log_sigma

flist <- alist(
  height ~ dnorm( mu, exp(log_sigma) ),
  mu ~ dnorm( 178, 20 ),
  log_sigma ~ dunif( 0, 50 )
)
m4.1_logsigma <- map( flist, data=d2 )
precis(m4.1_logsigma)

########################################

plot( d2$height ~ d2$weight )
m4.3 <- map (
  alist(
    height ~ dnorm( mu, sigma ),
    mu <- a + b*weight,
    a ~ dnorm( 156, 100 ),
    b ~ dnorm( 0, 10 ),
    sigma ~ dunif(0,50)
  ),
  data=d2
)
precis(m4.3, corr=TRUE)
post <- extract.samples(m4.3, n=1e4)
head(post)
plot(post)
dens(post$sigma,norm.comp = TRUE)

########################################

# data centering

d2$weight.c <- d2$weight - mean(d2$weight)
m4.4 <- map (
  alist(
    height ~ dnorm( mu, sigma ),
    mu <- a + b*weight.c,
    a ~ dnorm( 156, 100 ),
    b ~ dnorm( 0, 10 ),
    sigma ~ dunif(0,50)
  ),
  data=d2
)
precis(m4.4, corr=TRUE)

########################################

# superimpose MAP values on data

plot( height ~ weight, data=d2)
coef(m4.3)
abline(a=coef(m4.3)["a"], b=coef(m4.3)["b"])
post <- extract.samples(m4.3)
post[1:5,]


########################################

# draw more possible a,b combination sampled from joint posterior dist
N <- 352 # 10 # 100
dN <- d2 [1:N,]
mN <- map (
  alist(
    height ~ dnorm( mu, sigma ),
    mu <- a + b*weight,
    a ~ dnorm( 178, 100 ),
    b ~ dnorm( 0, 10 ),
    sigma ~ dunif( 0, 50 )
  ),
  data=dN
)
post <- extract.samples(mN, n=20)
plot( height ~ weight, data=dN,
      xlim=range(d2$weight), ylim=range(d2$height),
      col=rangi2, xlab="weight", ylab="height" )
mtext(concat("N=",N))
for( i in 1:N ) 
  abline( a=post$a[i], b=post$b[i], col=col.alpha("black",0.3))

########################################

# prediction

mu_at_50 = post$a + post$b * 50
dens(mu_at_50, norm.comp = TRUE, col=rangi2, lwd=2, xlab="mu|weight=50")
HPDI(mu_at_50, prob=0.89)

########################################

# link function

weight.seq <- seq( from=25, to=70, by=1 )
mu <- link( m4.3, data=data.frame(weight=weight.seq))
str(mu)

plot( height ~ weight, d2, type="n" )
for( i in 1:100 )
  points(weight.seq, mu[i,], pch=16, col=col.alpha(rangi2,0.1))

# summarize
mu.mean <- apply( mu, 2, mean )
mu.HPDI <- apply( mu, 2, HPDI, prob=0.89 )

plot( height ~ weight, d2, col=col.alpha(rangi2,0.5))
lines(weight.seq, mu.mean)
shade(mu.HPDI, weight.seq)


