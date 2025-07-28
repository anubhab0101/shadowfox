import time
from typing import Optional
from datetime import datetime, timedelta
import threading

class RateLimiter:
    """Rate limiter to control API call frequency"""
    
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute  # Minimum seconds between calls
        self.last_call_time = 0.0
        self.lock = threading.Lock()
        
        # Track call history for more sophisticated rate limiting
        self.call_history = []
        self.max_history_size = calls_per_minute * 2  # Keep twice the limit for buffer
    
    def wait_if_needed(self) -> float:
        """
        Wait if necessary to respect rate limits.
        Returns the time waited in seconds.
        """
        with self.lock:
            current_time = time.time()
            
            # Clean old entries from call history
            self._clean_call_history(current_time)
            
            # Check if we need to wait based on minimum interval
            time_since_last = current_time - self.last_call_time
            wait_time = 0.0
            
            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                time.sleep(wait_time)
                current_time = time.time()
            
            # Check if we're exceeding calls per minute
            if len(self.call_history) >= self.calls_per_minute:
                # Wait until the oldest call is more than a minute old
                oldest_call = self.call_history[0]
                time_to_wait = 60.0 - (current_time - oldest_call)
                
                if time_to_wait > 0:
                    time.sleep(time_to_wait)
                    wait_time += time_to_wait
                    current_time = time.time()
                    self._clean_call_history(current_time)
            
            # Record this call
            self.last_call_time = current_time
            self.call_history.append(current_time)
            
            return wait_time
    
    def _clean_call_history(self, current_time: float) -> None:
        """Remove calls older than 1 minute from history"""
        cutoff_time = current_time - 60.0
        self.call_history = [call_time for call_time in self.call_history 
                           if call_time > cutoff_time]
        
        # Limit history size to prevent memory issues
        if len(self.call_history) > self.max_history_size:
            self.call_history = self.call_history[-self.calls_per_minute:]
    
    def get_current_rate(self) -> float:
        """Get current calls per minute rate"""
        with self.lock:
            current_time = time.time()
            self._clean_call_history(current_time)
            return len(self.call_history)
    
    def get_time_until_next_call(self) -> float:
        """Get time in seconds until next call is allowed"""
        with self.lock:
            current_time = time.time()
            self._clean_call_history(current_time)
            
            # Check minimum interval
            time_since_last = current_time - self.last_call_time
            min_wait = max(0, self.min_interval - time_since_last)
            
            # Check rate limit
            rate_limit_wait = 0.0
            if len(self.call_history) >= self.calls_per_minute:
                oldest_call = self.call_history[0]
                rate_limit_wait = max(0, 60.0 - (current_time - oldest_call))
            
            return max(min_wait, rate_limit_wait)
    
    def can_make_call(self) -> bool:
        """Check if a call can be made immediately without waiting"""
        return self.get_time_until_next_call() == 0
    
    def reset(self) -> None:
        """Reset rate limiter state"""
        with self.lock:
            self.last_call_time = 0.0
            self.call_history = []
    
    def set_rate(self, calls_per_minute: int) -> None:
        """Update the rate limit"""
        with self.lock:
            self.calls_per_minute = calls_per_minute
            self.min_interval = 60.0 / calls_per_minute
            self.max_history_size = calls_per_minute * 2
            
            # Clean history if new rate is lower
            current_time = time.time()
            self._clean_call_history(current_time)
            if len(self.call_history) > calls_per_minute:
                self.call_history = self.call_history[-calls_per_minute:]

class AdaptiveRateLimiter(RateLimiter):
    """Rate limiter that adapts based on API response patterns"""
    
    def __init__(self, initial_calls_per_minute: int = 60):
        super().__init__(initial_calls_per_minute)
        self.success_count = 0
        self.error_count = 0
        self.consecutive_errors = 0
        self.last_adaptation = time.time()
        self.adaptation_interval = 60.0  # Adapt every minute
        
        # Rate adjustment parameters
        self.min_rate = 10  # Minimum calls per minute
        self.max_rate = 120  # Maximum calls per minute
        self.increase_factor = 1.1  # Factor to increase rate on success
        self.decrease_factor = 0.7  # Factor to decrease rate on error
    
    def record_success(self) -> None:
        """Record a successful API call"""
        with self.lock:
            self.success_count += 1
            self.consecutive_errors = 0
            self._maybe_adapt_rate()
    
    def record_error(self, is_rate_limit_error: bool = False) -> None:
        """Record a failed API call"""
        with self.lock:
            self.error_count += 1
            self.consecutive_errors += 1
            
            # Immediately reduce rate for rate limit errors
            if is_rate_limit_error:
                new_rate = max(self.min_rate, 
                             int(self.calls_per_minute * self.decrease_factor))
                self.set_rate(new_rate)
                self.last_adaptation = time.time()
            else:
                self._maybe_adapt_rate()
    
    def _maybe_adapt_rate(self) -> None:
        """Adapt rate based on success/error patterns"""
        current_time = time.time()
        
        if current_time - self.last_adaptation < self.adaptation_interval:
            return
        
        total_calls = self.success_count + self.error_count
        if total_calls < 5:  # Need minimum samples
            return
        
        success_rate = self.success_count / total_calls
        
        # Adjust rate based on performance
        if success_rate > 0.95 and self.consecutive_errors == 0:
            # High success rate, try increasing
            new_rate = min(self.max_rate, 
                         int(self.calls_per_minute * self.increase_factor))
        elif success_rate < 0.8 or self.consecutive_errors >= 3:
            # Low success rate or consecutive errors, decrease
            new_rate = max(self.min_rate, 
                         int(self.calls_per_minute * self.decrease_factor))
        else:
            new_rate = self.calls_per_minute
        
        if new_rate != self.calls_per_minute:
            self.set_rate(new_rate)
        
        # Reset counters
        self.success_count = 0
        self.error_count = 0
        self.last_adaptation = current_time
    
    def get_stats(self) -> dict:
        """Get rate limiter statistics"""
        with self.lock:
            return {
                'current_rate': self.calls_per_minute,
                'success_count': self.success_count,
                'error_count': self.error_count,
                'consecutive_errors': self.consecutive_errors,
                'success_rate': self.success_count / max(self.success_count + self.error_count, 1),
                'calls_in_last_minute': len(self.call_history),
                'time_until_next_call': self.get_time_until_next_call()
            }

def rate_limit_decorator(calls_per_minute: int = 60):
    """Decorator to add rate limiting to functions"""
    limiter = RateLimiter(calls_per_minute)
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            limiter.wait_if_needed()
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Global rate limiter instance for convenience
default_limiter = RateLimiter(calls_per_minute=30)  # Conservative default

def wait_for_rate_limit():
    """Convenience function to wait for rate limit using default limiter"""
    return default_limiter.wait_if_needed()
