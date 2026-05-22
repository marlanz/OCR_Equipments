import time
from fastapi import Request, HTTPException, status
from api.core.config import settings

class RateLimiter:
    """
    A lightweight, in-memory sliding window rate limiter.
    """
    def __init__(self, limit: int, period: int):
        self.limit = limit
        self.period = period
        self.requests = {}

    async def __call__(self, request: Request):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        
        if client_ip not in self.requests:
            self.requests[client_ip] = []
            
        # Remove requests older than the sliding window period
        self.requests[client_ip] = [t for t in self.requests[client_ip] if now - t < self.period]
        
        if len(self.requests[client_ip]) >= self.limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )
            
        self.requests[client_ip].append(now)

rate_limiter = RateLimiter(settings.RATE_LIMIT_LIMIT, settings.RATE_LIMIT_PERIOD)
