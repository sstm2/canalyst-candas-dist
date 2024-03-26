"""
Exceptions for candas
"""


class BaseCandasException(Exception):
    """
    Base Candas Exception class
    """


class ScenarioException(BaseCandasException):
    """
    Bad Scenario
    """


class CSINEmptyException(BaseCandasException):
    """
    Empty CSIN
    """


class AWSCredentialException(BaseCandasException):
    """
    Wrong input AWS Credentials
    """


class HTTPClientException(BaseCandasException):
    """
    Wrong input Canalyst API
    """
