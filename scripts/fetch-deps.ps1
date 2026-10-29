# Clone the sibling dependency layout that nimbus-edge builds against:
#
#   workspace\
#     nimbus-edge\       <- this repository
#     nimbus-infer-rs\   <- path dependency (..\nimbus-infer-rs)
#
# Run from the directory that will contain both checkouts.
git clone https://github.com/EAnge1o/nimbus-infer-rs.git nimbus-infer-rs
Write-Host "done: nimbus-infer-rs is now a sibling of nimbus-edge"
