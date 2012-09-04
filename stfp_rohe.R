require("MCMCpack")
require("mclust")

## Sample an undirected graph on n vertices
## Input: P is n times n matrix giving the parameters of the Bernoulli r.v.
rg.sample <- function(P){
  n <-  nrow(P)
  U <- matrix(0, nrow = n, ncol = n)
  U[col(U) > row(U)] <- runif(n*(n-1)/2)
  U <- (U + t(U))
  A <- (U < P) + 0 ;
  diag(A) <- 0
  return(A)
}

## Sample a SBM graph
## Input:
## n is number of vertices
## B is the K times K block probability matrix
## rho is a vector of length K giving the categorical
## distribution for the memberships
## Output:
## adjacency is the n times n adjacency matrix
## tau is the membership function
rg.SBM <- function(n, B, rho){
  tau <- sample(c(1:length(rho)), n, replace = TRUE, prob = rho)
  P <- B[tau,tau]
  return(list(adjacency=rg.sample(P),tau=tau))
}

nonpsd.laplacian <- function(A){

    n = nrow(A)
    s <- rowSums(A)
    L <- diag(s)/(n-1) + A

    return(L)
}

svd.extract <- function(A, dim = NULL){

    L <- nonpsd.laplacian(A)
    L.svd <- svd(L)

    if(is.null(dim))
      dim <- scree.thresh(L.svd$d)
    
    L.svd.values <- L.svd$d[1:dim]
    L.svd.vectors <- L.svd$v[,1:dim]

    if(dim == 1)
      L.coords <- L.svd.values * L.svd.vectors
    else
      L.coords <- L.svd.vectors %*% diag(L.svd.values)

    return(L.coords)
}

## Perform stfp embedding. Use mclust to cluster the embedded points
## Input:
## A is adjacency matrix
## dim is the dimension to embed to
## G is the number of clusters
inverse.rdpg <- function(A, dim, G = NULL){
  if(is.list(A)){
    for(i in 1:length(A)){
      if(i == 1){
        X <- svd.extract(A[[i]], dim)
      }
      else{
        X <- cbind(X, svd.extract(A[[i]], dim))
      }
    }
  }
  else{
    X <- svd.extract(A,dim)
  }

  X.mclust <- Mclust(X, G)
  return(list(X = X, cluster = X.mclust))
}

rohe.normalized.laplacian <- function(W){
    S <- apply(W,1,sum)
    return(diag(1/sqrt(S)) %*% W %*% diag(1/sqrt(S)))
}

## Perform the Laplacian embedding of Rohe, Chatterjee & Yu
rohe.laplacian.map <- function(A, dim, G = NULL, scaling=FALSE){
  
  L <- rohe.normalized.laplacian(A)
  decomp <- eigen(L,symmetric=TRUE)

  ## Use the dim largest eigenvalues in absolute value
  decomp.sort <- sort(abs(decomp$values), decreasing = TRUE, index.return = TRUE)
  eigen.vals <- decomp$values[decomp.sort$ix[1:dim]]
  eigen.vectors <- decomp$vectors[,decomp.sort$ix[1:dim]]
  
  if(scaling){
    Psi <- eigen.vectors * outer(seq(1,1,length.out = nrow(L)),eigen.vals)
  }
  else{
    Psi <- eigen.vectors
  }
  Psi.mclust <- Mclust(Psi, G)
  
  return(list(X=Psi,cluster=Psi.mclust))
}

stfp.laplacian.experiment <- function(nmc){
  results <- matrix(0,nrow = nmc, ncol = 3)
  n <- 1000
  rho <- c(0.6,0.4)
  B <- matrix(c(0.42,0.42,0.42,0.5),nrow = 2,ncol=2)
  for(i in 1:nmc){
    A <- rg.SBM(n,B,rho)
    stfp.embed <- inverse.rdpg(A$adjacency, dim = 2, G = 2)
    rohe.embed <- rohe.laplacian.map(A$adjacency, dim = 2, G = 2)
    aaa <- stfp.laplacian.mcnemar.test(stfp.embed$cluster$classification,
                                       rohe.embed$cluster$classification, A$tau)
    results[i,] = c(aaa$L1,aaa$L2,aaa$pval)
  }
  return(results)
}
    
## Warning: Very non-robust.
## Assume g1 and g2 are categorical in {1,2}
stfp.laplacian.mcnemar.test <- function(g1, g2, y){
  tmp1 <- sum( abs(g1 - y) > 0)/length(g1)
  if(tmp1 > 0.5)
    g1 <- 3 - g1
  
  tmp1 <- sum( abs( g2 - y) > 0)/length(g2)
  if(tmp1 > 0.5)
    g2 <- 3 - g2

  T <- matrix(0,nrow = 2, ncol = 2)
  ttt1 <- abs(g1 - y) > 0
  ttt2 <- (abs(g2 - y) > 0)
  T[1,1] <- sum(ttt1*ttt2)
  T[1,2] <- sum((1 - ttt1)*ttt2)
  T[2,1] <- sum(ttt1*(1 - ttt2))
  T[2,2] <- sum((1-ttt1)*(1 - ttt2))
  return(list(L1 = sum(ttt1)/length(ttt1),
              L2 = sum(ttt2)/length(ttt2),
              pval=mcnemar.test(T)$p.value))
}
