package sutradhara.main

import future.keywords.if

# Tenant-scoped dispatch: a tenant's policies live under
# data.sutradhara.tenants.<tenant>.main, loaded from whatever extra root
# path OPA was started with (see EXTRA_POLICY_PATH in run.sh/entrypoint.sh).
# Delegating to the tenant's own "main" package (rather than indexing
# data.sutradhara.tenants[tenant][role] directly here) lets each tenant own
# its own role-name mapping — e.g. myPatasala's RoleEnum is upper-cased
# ("ADMIN", "PARENT", ...) while this repo's demo ./roles/ convention is
# lower-cased ("admin", "parent", ...). This file never references a tenant
# by name — adding a new tenant's policy folder requires zero edits here.
decision := data.sutradhara.tenants[input.tenant].main.decision if {
    input.tenant
}

# No tenant specified: fall back to the generic, role-only demo policies
# bundled in ./roles/ (unchanged, backward compatible).
decision := data.sutradhara.roles[input.user.role].decision if {
    not input.tenant
}

# Default deny for any unrecognised/missing tenant or role
default decision = {
    "authorized": false,
    "columns": [],
    "filter": "1=0",
    "error": "Invalid or missing tenant/role"
}
