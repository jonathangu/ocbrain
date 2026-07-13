# Operations artifacts

OCBrain v1 installs only the on-demand local MCP launcher. It does not install
recurring launchd work. The inert files here preserve the v0.4.1 retirement
decision for operators upgrading from a legacy installation.

The three `com.jonathangu.ocbrain.*.plist` files are retired identifiers kept
only to help operators find and remove legacy installations. They are disabled,
contain no timer, and execute only `/usr/bin/false`. Do not install or load them.

The `autopilot` and stall-diagnostic code may still be invoked manually for an
owned maintenance operation. A manual invocation is an explicit extra; it is
not part of MCP startup, runtime synchronization, or the default install.
