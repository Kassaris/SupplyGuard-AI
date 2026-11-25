"""Who may do what (services/identity.py), with the users from users.csv."""

import pytest

from supplyguard_ai.config import get_settings
from supplyguard_ai.services.identity import AuthorizationError, IdentityService

PROCUREMENT_MANAGER = "u-pm-01"
ANALYST = "u-an-01"
COORDINATOR = "u-ic-01"
FINANCE_CONTROLLER = "u-fc-01"
VIEWER = "u-ro-01"


@pytest.fixture(scope="module")
def identity() -> IdentityService:
    return IdentityService.from_csv(get_settings().support_data_dir / "config" / "users.csv")


def test_unknown_user_is_rejected(identity):
    with pytest.raises(AuthorizationError):
        identity.authenticate("u-nobody")


def test_only_finance_controller_reads_finance_documents(identity):
    for user in identity.all_users():
        allowed = identity.can_read_document(user, "finance-confidential")
        assert allowed == (user.user_id == FINANCE_CONTROLLER)


def test_viewer_reads_only_public_documents(identity):
    assert identity.allowed_document_scopes(identity.authenticate(VIEWER)) == {"public"}


def test_viewer_cannot_open_a_case(identity):
    with pytest.raises(AuthorizationError):
        identity.require_can_open_case(identity.authenticate(VIEWER))


@pytest.mark.parametrize("user_id", [PROCUREMENT_MANAGER, ANALYST, COORDINATOR])
def test_operational_roles_can_submit_plans_for_approval(identity, user_id):
    identity.require_can_request_approval(identity.authenticate(user_id))  # must not raise


@pytest.mark.parametrize("user_id", [FINANCE_CONTROLLER, VIEWER])
def test_other_roles_cannot_submit_plans_for_approval(identity, user_id):
    with pytest.raises(AuthorizationError):
        identity.require_can_request_approval(identity.authenticate(user_id))


def test_user_cannot_sign_for_another_role(identity):
    manager = identity.authenticate(PROCUREMENT_MANAGER)
    identity.require_can_approve_as(manager, "procurement_manager")  # own role: allowed
    with pytest.raises(AuthorizationError):
        identity.require_can_approve_as(manager, "finance_controller")
