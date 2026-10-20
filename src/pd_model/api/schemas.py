from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictInt

Money = Annotated[FiniteFloat, Field(ge=0)]
Status = Annotated[StrictInt, Field(ge=-2, le=9)]


class ClientData(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    LIMIT_BAL: Annotated[FiniteFloat, Field(gt=0)]
    SEX: Annotated[StrictInt, Field(ge=1, le=2)]
    EDUCATION: Annotated[StrictInt, Field(ge=0, le=6)]
    MARRIAGE: Annotated[StrictInt, Field(ge=0, le=3)]
    AGE: Annotated[StrictInt, Field(ge=18, le=100)]
    PAY_0: Status
    PAY_2: Status
    PAY_3: Status
    PAY_4: Status
    PAY_5: Status
    PAY_6: Status
    BILL_AMT1: FiniteFloat
    BILL_AMT2: FiniteFloat
    BILL_AMT3: FiniteFloat
    BILL_AMT4: FiniteFloat
    BILL_AMT5: FiniteFloat
    BILL_AMT6: FiniteFloat
    PAY_AMT1: Money
    PAY_AMT2: Money
    PAY_AMT3: Money
    PAY_AMT4: Money
    PAY_AMT5: Money
    PAY_AMT6: Money


class PredictionResponse(BaseModel):
    default_prediction: Annotated[StrictInt, Field(ge=0, le=1)]
    default_probability: Annotated[FiniteFloat, Field(ge=0, le=1)]
    threshold: Annotated[FiniteFloat, Field(ge=0, le=1)]
    model_version: str
    schema_version: str
