mkdir -p ${DISC_DATA}/{query,reference,train}_images
mkdir -p ${DISC_DATA}/query_images_phase2

# ref set
cd ${DISC_DATA}/reference_images
wget -O 11.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An9HmE5IdG3f1BLhSzy6onve1VIjUNn0-7cLRHjJTtJZujlE4heVuJbrwe2gk2IwlYRb00LoxksgbiDF6oQmbT61LLjT2m-oo6ZQkeC30e5O.zip?ccb=10-5&oh=00_AfAX_5w_JY6anV5yuDoE4xoepD7ztosfiPQlX_Gb1ZWPMw&oe=64060FFF&_nc_sid=387328"
wget -O 12.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An9peGKIJR3mqkGAxf1ZKG-tTqgfyGr5k4d-0x9mBDzkaetfQDaokL2RvrHdIgn3QnVRHiaisKg9yigXQWzla2yW6Ow-hatsqvcNXugiE6vF.zip?ccb=10-5&oh=00_AfD24bBWYLSwwHDERgKCtEVJtGiqNqjt35KHbSad0gS4hg&oe=64060EC6&_nc_sid=387328"
wget -O 13.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An8PWol-yJUKw0jdO7oLFrIJLadCWlSyQij5bu6NnOQlrSvsRH8NzM0-Z2nVeJIkz-hz9o8vkj6QvvLmxix0GXetU36xE_bq9btTMmpy4RXF.zip?ccb=10-5&oh=00_AfAKkWTNOQt1sa4soyegTyU94Pb5xcvFFoq-FgDSxlyz_g&oe=6405FF78&_nc_sid=387328"
wget -O 14.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An8oLy3MDGArAjjmOCfD--IwE3SRV0Ve721BKimAtZ1bv19UaHAj71x0veCbSZcQ5kwCFd08mbZavIJNi8uWfIuOrNYc2KZxcYkPU2-9iygS.zip?ccb=10-5&oh=00_AfDjC5Z4mR-Xb3rSCfOK8DbTVjx5sMrRFsFMUw9csG7nYg&oe=6405F31F&_nc_sid=387328"
wget -O 15.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An8T6HDDnTCb_FBzTeEkBH25GPc1jbFDrgP-zcDwYw6uoNZcALrFIlXS580GNbfIaslmC8EZ8eLfqUIgnqEc9TpfL98Lg55NhdUe5BL2Xmmn.zip?ccb=10-5&oh=00_AfD4BGUzVPjaW6WdvLz3tiGj-m64s35Ybus_0jxWFJaJqA&oe=6405E3A8&_nc_sid=387328"
wget -O 16.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An9gidsv4LdQWLx987MzpUfsts0Y0mvutW6w-03tZ0tZLLG3YXm_Gj3FYE1RQKqisKvCZSRliIrlXgZoukdigR-BDgCerO9vzXR1MNBHm723.zip?ccb=10-5&oh=00_AfDx7m5n3nUqS15baFNzYUH_4OpyiG1Fi7JkaKzywj2ImA&oe=640610B7&_nc_sid=387328"
wget -O 17.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An_lQCsxCPcyL-ZkfFtTKfC-vIJwxarSq8ZxmWzguI1eknyBJAVmyM176uR4j_k4Mof8WG9NdnLpaxv3HHtUlidUdzS3kTW9COpZ9cU24pgB.zip?ccb=10-5&oh=00_AfAiAzG7qkFqCQi_2bIguck2Ih6bwG2HcUTQuJHAClo2Tw&oe=6405FE33&_nc_sid=387328"
wget -O 18.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An8wbmDUcR02PiforPMK51geWyo8w-Z1Xb9HJUhY40l0lpoDHZoGcTe58hK__l5EdMfhQZFA60ExsHi05bVG6cv_kpz7wIb3DgZWpgMj3mYC.zip?ccb=10-5&oh=00_AfBB6c6l8RC5d3DV0ptRrGyJx0RJOGg11QRqULjLf14s7w&oe=64060C3A&_nc_sid=387328"
wget -O 19.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An8eobQmHvBmwm-WngSd9s8YZN4zrrBbVTEM-oHxPf-6WPqM70ULgUdKb1M-g0Ee3CEWOxOEu8A-dWzr9Tq6vO7uGoyk_yJli2NlK5dN8sYT.zip?ccb=10-5&oh=00_AfA1bHyHdTU7tk7EbAHi0cRAZtwrqtwefDuvrubK_i7Wcw&oe=64060D8B&_nc_sid=387328"
wget -O 2.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An_7WGYlE7jejObaK40MIcNvASAhwYG0xY_UTmhHdmK98v1kB2eo83IgMvbc4aiNBFLPdghUfJITw7U-MtjRec607x78fPeYGnWo6m3a36A.zip?ccb=10-5&oh=00_AfAfBVJ2LrS6VkcNOHpjmRuppVpeld1DVbkcKoqdlRPZ2g&oe=64061441&_nc_sid=387328"
wget -O 3.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An_BK-CJQoI9DH1FpMZfmzLolWI1_XcmHlvGNv8RL7X_itivPBd2Hi9LlEY9K8voOstYE9bzeGgckF28C3kQyvPkgrn5xiS4ImFzrmMgDuI.zip?ccb=10-5&oh=00_AfDI-AW-xsOZoDc-HAxLDmQAA4H0L6VfMtvyCPyG8-sLqA&oe=640602D3&_nc_sid=387328"
wget -O 4.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An9tIZcNxFDpvP_hOoLNtVu_4_a7xjBoF7YnK2OZaFzd-A5Ozm154Djhr670nZR62YVQbrI9atn1bvv-rN_DcOwt6YuUM0ILVefAnIt27WU.zip?ccb=10-5&oh=00_AfAbXoAIqJhcKeCmX4suTJ2xt-zceL83ozpjjRNBpK15wg&oe=64060480&_nc_sid=387328"
wget -O 5.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An-pxjcfCiobr1nxS7h9gK-UUodgUuaoLYyH3D_d5PxO54ExfHWAuEnjyVl6J9IQrXJypFgO0ggfyyWIhA-VLDPBm4XDp6mU32ac0n_X5Y8.zip?ccb=10-5&oh=00_AfAIFr6rguYgdJTGMFCVs0LtrgUwsu_7d5dKhZBZu4bfMA&oe=6405F6B6&_nc_sid=387328"
wget -O 6.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An-oTy3QWn19e_TaLy01Gn-qetHfUS5qvGn3V3jfAi_ihgFdqIJikC1y-X0DABGzBhR9b2VRHZFfxSO4VcwusOAy_yDgdB2yit4_4-2nUb4.zip?ccb=10-5&oh=00_AfCLueZ8H9szVkl-rm6SGp_j3Ruxi3fItHII93tYrkBeAQ&oe=6406026C&_nc_sid=387328"
wget -O 7.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An-SGPQnsK4IefSvuO-7_4MnRBft8YSSidlp2CDwTCN5LYhvkA9L5NYIZ8Iltz7zTAQ-06mWXb-XwGncq8h1yxDvCBbhx2nTs3E7VBJaMAY.zip?ccb=10-5&oh=00_AfBxg07ZhuAJ7-apVEN6xnDAvck5BwXSduOUTDdpQGKYBQ&oe=6405E373&_nc_sid=387328"
wget -O 8.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An9JKZ53qfWPCXOIfWvDQeeV83szo3CyHPFZZIVdx8NnEBeFM-EcyJDPiQqhHXjqfAUYqUCCAtr3TV7Gw2mZFtCPcq3N-URW5ZOSaKLah3E.zip?ccb=10-5&oh=00_AfDV86OTijPWnGVt_GR0IaE1ZgzpuhrbKtb7DL7Y9-HG4Q&oe=6405E3C7&_nc_sid=387328"
wget -O 9.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An_KPT-RcKgH7471Omw1P2EZO9YOFqu9Taee9VlzmTdzbbrR-ScnIRdSWPQ_n3K9ftHc8MQrI5dLmOvMAnuNj12_t9y6W9IZBqWud93asNw.zip?ccb=10-5&oh=00_AfCi60fYL7wX3wbGkcQixO74SDwxY-RKpGewRb9sK49rMA&oe=6405EA2B&_nc_sid=387328"

for i in {0..19}; do
    unzip ${i}.zip -d ${i}/
done


# train set
cd ${DISC_DATA}/train_images
wget -O 0.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An9Bmbe9vm85YAuM2kVgHyVfjBCwSkZIGIBHD4jmgNl-5sanAWdX6zHqD52QOqHuvDsaQpYvD-sr8vgH2MGWiCcX__B6rr2-kVEK24UniHa5dQ.zip?ccb=10-5&oh=00_AfBCcV384BCEP22QwlstY2fBt7Rc0wAN90vStAfwKVUHtA&oe=640604D4&_nc_sid=387328"
wget -O 1.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An81t51hFtVuUh1wzRrfqAGrhe1Q8HGpqioAGA-5nI8sZ9wzsjRlpLGkAUasTL7ItTWgUmMdWrteioJniDsDdNDRJUENqiOz94WKd0kGCbeIcQ.zip?ccb=10-5&oh=00_AfCQHNUMjixvExP735JYZ4GFSN3MrcownQdeSdq5uO2nLQ&oe=6405E16D&_nc_sid=387328"
wget -O 10.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An9hKI1Juq8IItQEEOs1K167mN92vdyYJ9Vjj53ZkD7TLTRq5Wn22Zff2aI1hTvs4-x2GVSe4-FXmRNNX6RJ3G3X068Jj7ucRoXaaanw-iXaEnI.zip?ccb=10-5&oh=00_AfDe6rhvgGo-wJ6pG_9K4l5rJCgjKk_mBSiy2BuRr-VmUQ&oe=6405F1CD&_nc_sid=387328"
wget -O 11.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An_67mEZiiC2HIcQxci0iP3oLFy_b9nQPtiQHYAYkT-EyvCrzV6zM-hj3X-Tq-HrIyJIn3FSMAYa2_lG95rBIOABUucIt0LPvaXZzWeCIMjZgqg.zip?ccb=10-5&oh=00_AfAn2upAvd0yJHlXwB0Wm_CVNZAeq-mUYbB8IXOBvZQoYw&oe=6405F51F&_nc_sid=387328"
wget -O 12.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An8JYiPTZUBbVMsk5dMMFnGDA6U9FcmcZvVYGZzxF8nA6iVXL5RTV4lJQftbrN6BP1B_G0W4I2JJMshd6ZtlvFGA-EarHf5og2gYPhPE-aDyors.zip?ccb=10-5&oh=00_AfB5-8S3YM30ihTv-aa407cYDxzJ2AwCmkaS1WD27JOkuQ&oe=6405F5AF&_nc_sid=387328"
wget -O 13.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An93fIlhJN9rsXmrH_4e8-OrHmhchZqytswCwX9tyX6W3H4-U8P2SICv_fy7RM-L9wWwZ-2AZBztxU4h04HUWbip4AULoDSSHNAHdkLIk-iRyV0.zip?ccb=10-5&oh=00_AfDhAr6nS_qH2ahRMUouX0X5ibJ4fLZcUVVx5b_H-Am64g&oe=64061097&_nc_sid=387328"
wget -O 14.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An926GkNZ5cgbPgpibq9cntljJf43oWDipzXfDwiRpM8NAkbuv7H-e_WRemcv0JRnKbg89CDIWrv8voW55hKrGWl0681xklVbK0YPSb05zySdA8.zip?ccb=10-5&oh=00_AfByg8x6O_G_3wxkvyziqpAH55A7o2EHBxaEDSJNnF6Oqg&oe=64061542&_nc_sid=387328"
wget -O 15.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An_cF1jHcUokNDXJI2Dec6kYFZtdjwjCi31h3TCe_psjv4Z9fhUQcn2mB99-MihoOMMBUciSPKsSMVCCmwM4XPty-VPmACmQTRY0B9_SwzAH048.zip?ccb=10-5&oh=00_AfDCf_Z9gNAW2WNqjPQbxbnrRHosNyeFfYLTvSxCpFNguQ&oe=6406028F&_nc_sid=387328"
wget -O 16.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An9y9wrKj2TGh2XcsGUJ-2Y7qxPBc0OyBOvmU-DVz9Y2DeBEDQPRfWgtA3XZQ_vkfClPM5U1aO17IFbTuUabwoMiTDW-VhmQAF8UCcyTumPbM2s.zip?ccb=10-5&oh=00_AfCTpKlnRpeULQXdTOlQTmELAkYgXt0KbqeNFPvbKK_GJQ&oe=6405E281&_nc_sid=387328"
wget -O 17.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An_S2wZrNZDf2ww6bB3GKI_2KG9pjKtFcIw3uAJw559OPHB-QBY2FygeEi-cmx4AWVnmf363Dpg-ino6PHqbtQDo7_9nPbCJ5_1p6StYbhKKBFE.zip?ccb=10-5&oh=00_AfA9FElBqT_jT0_U06OFiV_4qAWDUz9WVPX9ZVZ0Q8AW2A&oe=6405FE0A&_nc_sid=387328"
wget -O 18.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An-goVwpDJfLoZbMWWRdV0ZumM2zNTxbLt4GiftOVydEbLUllqI64FKGTa-fRtYThkSEmbgU54SJet3KHoPSJAhGXzTI-k9Z8C5Qbr-NH3K_7OE.zip?ccb=10-5&oh=00_AfCP10kWTHaO2LDCLqmWgPhBfrUBxTHXchBcvy6W6vmW6A&oe=640611CA&_nc_sid=387328"
wget -O 19.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An-_RtDDeTsVvoMJF4r2Ek6JifYpNGcVKAthkkQlcQ2MXTIB7o40PvuubkluswCPPBwf2hrXihbvEx4OIiiH9oME7PwuC1CL4MyDCDdlzGZfp7Y.zip?ccb=10-5&oh=00_AfDXGpWo3V1UCEpr9vmZik7XdmQkSmvkfmnO1fHNiumsaw&oe=640610F8&_nc_sid=387328"
wget -O 2.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An-MA6YBihJdQpYXMKn5FKKlTyaJ6B2rVjczFc5wY8M6GZm5qvBb-SFfIFM1Em30Ml9ZFkYHZjruUp4VbyLeTtG2FOiJzs1XAEIvMqEKlO6ddA.zip?ccb=10-5&oh=00_AfAs9HKiqp8NwGylsuMYHgEKgfd93UzdyAiLNKEljazQvA&oe=640604D1&_nc_sid=387328"
wget -O 3.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An_N3Oomj8X2yuexaX9N6qgJ7XMxVNIuzlYbKONWPJtlEX_Id1bjKk0gHVH_9makWul_Y1SD7HEgCGUtQ_CaYR6oSGVNDWt54kSMuvRJHyRC_A.zip?ccb=10-5&oh=00_AfCUwJTi1qpraL1xyKWGU5DMpNHoMs-xEHM_ZfsGis7lNA&oe=640613D9&_nc_sid=387328"
wget -O 4.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An8TQkuXrWqBJdd3fjxruSNjyMQpgn20G1ZNWR6ydEeOPj7vC7QuOhv0Bx3jQfFi0QHxCBhSIEW8PgARFgAEgmge4lu0LOOyxU9ME4ak-KJ4YA.zip?ccb=10-5&oh=00_AfBBjkEKE9Chbk7p5SiBIhqt_kfFK1ZB1aw0PatJKIRu4Q&oe=6405F5CE&_nc_sid=387328"
wget -O 5.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An_jQ3L4tlJf8s7eVzqIeujfBpdYKmha_xtZtp6ADxkYRLrlYL-6_Ji_SNNQp9jq39i6ra5huVrDMMX-4t3xA6na_8Kl1XDR8sOAOZF-POXahw.zip?ccb=10-5&oh=00_AfC6M1YWLLOBY2v_InZrgoCPXbO9yQ_y2qXpXyocK9SgGg&oe=64060110&_nc_sid=387328"
wget -O 6.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An9dJpzFim8AmxvwxC4N94SVQaYBCFo1qfq4rUyIITXtakrBysxAWV8eMzMDF-JHJ1h13QAAbCQg2CAACNnRffzhpasuz7njPdVb_tv1RvvUOw.zip?ccb=10-5&oh=00_AfDsh-ENRikK1mqy5gpcnIVuMyVqFOUN-lxyBam2l_6__w&oe=6405E173&_nc_sid=387328"
wget -O 7.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An90LzLa0W8hUPLk40tdD0_PnHUjafpdR0wxuYYvEf_1O1wjo5x3MwnMcGn5KMJrXEJoCsJVhnb00IgwDozkFV_x9BEJ6VxKsO4-MVuhIpnPfQ.zip?ccb=10-5&oh=00_AfAYY1BoX7y_zHZi_UPg-ysBvr2xVi-TiCYXjVttAP0pCA&oe=6405E797&_nc_sid=387328"
wget -O 8.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An9cmVtphS4ejn6QX0wbsknMGpqYFUCdxEhVqOrDLzEyKL6j8MMrcDWhuWerQbCpSuwkWgMukI97rS5YBDh8gSX9axQPs8HDH2j9si-X-xd51w.zip?ccb=10-5&oh=00_AfCl99i1OAXlmV_nXwBiqWEU4uRyK53qWhVvyDcotoUtVA&oe=64060C97&_nc_sid=387328"
wget -O 9.zip "https://scontent-nrt1-1.xx.fbcdn.net/m1/v/t6/An8H1TO8TBEJv3DYmKnEuu8aVxsEEaUrza5QdDHAGL_jLQYXFyxzl1tuFeMnz2BA4a6Q4wwryfmNTwx326aE9xv5sU0Gdze7lJjka6EiQcIO3w.zip?ccb=10-5&oh=00_AfAv_G_YHz5NYT4Az-dAd2nuWRynqUvTCSyb2SZV3wGk5A&oe=6406097F&_nc_sid=387328"

for i in {0..19}; do
    unzip ${i}.zip -d ${i}/
done
