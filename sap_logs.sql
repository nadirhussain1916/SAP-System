PGDMP                      }            sap_logs    17.4    17.4     �           0    0    ENCODING    ENCODING        SET client_encoding = 'UTF8';
                           false            �           0    0 
   STDSTRINGS 
   STDSTRINGS     (   SET standard_conforming_strings = 'on';
                           false            �           0    0 
   SEARCHPATH 
   SEARCHPATH     8   SELECT pg_catalog.set_config('search_path', '', false);
                           false            �           1262    2815541    sap_logs    DATABASE     �   CREATE DATABASE sap_logs WITH TEMPLATE = template0 ENCODING = 'UTF8' LOCALE_PROVIDER = libc LOCALE = 'English_United States.1252';
    DROP DATABASE sap_logs;
                     postgres    false            �            1259    2815622 
   logs_fixed    TABLE     �  CREATE TABLE public.logs_fixed (
    started time without time zone,
    server text,
    program text,
    work_process integer,
    "user" text,
    response_time_in_ms integer,
    time_in_work_process_ms integer,
    wait_time_ms integer,
    cpu_time_ms integer,
    db_request_time_ms integer,
    vmc_elapsed_ms integer,
    enqueue_time_ms integer,
    enqueues integer,
    program_load_time_ms integer,
    screen_load_time_ms integer,
    load_time integer,
    roll_ins integer,
    roll_outs integer,
    roll_in_time_ms integer,
    roll_out_time_ms integer,
    roll_wait_time_ms integer,
    number_of_roundtrips integer,
    direct_read_requests integer,
    direct_read_database_rows integer,
    direct_read_buffer_requests integer,
    direct_read_request_time_ms integer,
    direct_read_average_time_rows_ms integer,
    sequential_read_request integer,
    sequential_read_database_rows integer,
    sequential_read_buffer_request integer,
    read_pysical_database_calls integer,
    sequential_read_request_time_ms integer,
    sequential_read_average_time_row_ms integer,
    update_requests integer,
    update_database_rows integer,
    update_pysical_database_calls integer,
    update_request_time_ms integer,
    update__average_time_rows integer,
    delete_requests integer,
    delete_database_rows integer,
    delete_physical_database_calls integer,
    delete_request_time_ms integer,
    delete__average_time_row_ms integer,
    insert_requests integer,
    insert_database_rows integer,
    insert_pysical_database_calls integer,
    insert_request_times_ms integer,
    insert__time_row_ms integer,
    work_process_number integer,
    maximum_memory_roll_kb integer,
    total_allocated_page_memory_kb integer,
    maximum_extended_memory_in_task_kb integer,
    maximum_extended_memory_in_step_kb integer,
    extended_memory_in_use_kb integer,
    privilege_memory_in_use_kb integer,
    work_progress_in_privilege_mode boolean
);
    DROP TABLE public.logs_fixed;
       public         heap r       postgres    false            �            1259    2815616    sap_logs    TABLE     �  CREATE TABLE public.sap_logs (
    "Started" text,
    "Server" text,
    "Program" text,
    "Work Process" bigint,
    "User" text,
    "Response Time in ms" text,
    "Time In Work Process ms" text,
    "Wait time ms" bigint,
    "CPU time ms" text,
    "DB request time ms" text,
    "VMC elapsed ms" bigint,
    "Enqueue time ms" bigint,
    "Enqueues" bigint,
    "Program Load time ms" bigint,
    "Screen load time ms" bigint,
    "Load time" bigint,
    "Roll ins" bigint,
    "Roll outs" bigint,
    "Roll in time ms" bigint,
    "Roll out time ms" bigint,
    "Roll wait time ms" text,
    "Number of roundtrips" bigint,
    "Direct read requests" text,
    "Direct read Database rows" text,
    "Direct read buffer requests" text,
    "Direct read request time ms" text,
    "Direct read average time/rows ms" text,
    "Sequential read request" text,
    "Sequential read database rows" text,
    "Sequential read buffer request" text,
    "Read pysical database calls" text,
    "Sequential read request time ms" text,
    "Sequential read average time/row ms" text,
    "Update requests" bigint,
    "Update database rows" text,
    "Update pysical database calls" bigint,
    "Update request time ms" bigint,
    "Update  average time/rows" text,
    "Delete requests" text,
    "Delete database rows" text,
    "Delete physical database calls" text,
    "Delete request time ms" bigint,
    "Delete  average time/row ms" text,
    "Insert requests" text,
    "Insert database rows" text,
    "Insert pysical database calls" text,
    "Insert request times ms" bigint,
    "Insert  time/row ms" text,
    "Work process number" bigint,
    "Maximum memory roll KB" bigint,
    "Total allocated page memory KB" bigint,
    "Maximum extended memory in task KB" text,
    "Maximum extended memory in step KB" text,
    "Extended memory in use KB" text,
    "Privilege memory in use KB" text,
    "Work progress in privilege mode" text
);
    DROP TABLE public.sap_logs;
       public         heap r       postgres    false            �            1259    2815605    sheet1    TABLE     �  CREATE TABLE public.sheet1 (
    id integer,
    started time without time zone,
    server text,
    program text,
    work_process integer,
    "user" text,
    response_time_in_ms bigint,
    time_in_work_process_ms bigint,
    wait_time_ms bigint,
    cpu_time_ms bigint,
    db_request_time_ms bigint,
    vmc_elapsed_ms bigint,
    enqueue_time_ms bigint,
    enqueues bigint,
    program_load_time_ms bigint,
    screen_load_time_ms bigint,
    load_time bigint,
    roll_ins bigint,
    roll_outs bigint,
    roll_in_time_ms bigint,
    roll_out_time_ms bigint,
    roll_wait_time_ms bigint,
    number_of_roundtrips bigint,
    direct_read_requests bigint,
    direct_read_database_rows bigint,
    direct_read_buffer_requests bigint,
    direct_read_request_time_ms bigint,
    direct_read_average_time_rows_ms double precision,
    sequential_read_request bigint,
    sequential_read_database_rows bigint,
    sequential_read_buffer_request bigint,
    read_pysical_database_calls bigint,
    sequential_read_request_time_ms bigint,
    sequential_read_average_time_row_ms double precision,
    update_requests bigint,
    update_database_rows bigint,
    update_pysical_database_calls bigint,
    update_request_time_ms bigint,
    update_average_time_rows double precision,
    delete_requests bigint,
    delete_database_rows bigint,
    delete_physical_database_calls bigint,
    delete_request_time_ms bigint,
    delete_average_time_row_ms double precision,
    insert_requests bigint,
    insert_database_rows bigint,
    insert_pysical_database_calls bigint,
    insert_request_times_ms bigint,
    insert_time_row_ms double precision,
    work_process_number bigint,
    maximum_memory_roll_kb bigint,
    total_allocated_page_memory_kb bigint,
    maximum_extended_memory_in_task_kb bigint,
    maximum_extended_memory_in_step_kb bigint,
    extended_memory_in_use_kb bigint,
    privilege_memory_in_use_kb bigint,
    work_progress_in_privilege_mode boolean
);
    DROP TABLE public.sheet1;
       public         heap r       postgres    false            �            1259    2815589 
   sheet1_old    TABLE     �  CREATE TABLE public.sheet1_old (
    "Started" text,
    "Server" text,
    "Program" text,
    "Work Process" bigint,
    "User" text,
    "Response Time in ms" text,
    "Time In Work Process ms" text,
    "Wait time ms" bigint,
    "CPU time ms" text,
    "DB request time ms" text,
    "VMC elapsed ms" bigint,
    "Enqueue time ms" bigint,
    "Enqueues" bigint,
    "Program Load time ms" bigint,
    "Screen load time ms" bigint,
    "Load time" bigint,
    "Roll ins" bigint,
    "Roll outs" bigint,
    "Roll in time ms" bigint,
    "Roll out time ms" bigint,
    "Roll wait time ms" text,
    "Number of roundtrips" bigint,
    "Direct read requests" text,
    "Direct read Database rows" text,
    "Direct read buffer requests" text,
    "Direct read request time ms" text,
    "Direct read average time/rows ms" text,
    "Sequential read request" text,
    "Sequential read database rows" text,
    "Sequential read buffer request" text,
    "Read pysical database calls" text,
    "Sequential read request time ms" text,
    "Sequential read average time/row ms" text,
    "Update requests" bigint,
    "Update database rows" text,
    "Update pysical database calls" bigint,
    "Update request time ms" bigint,
    "Update  average time/rows" text,
    "Delete requests" text,
    "Delete database rows" text,
    "Delete physical database calls" text,
    "Delete request time ms" bigint,
    "Delete  average time/row ms" text,
    "Insert requests" text,
    "Insert database rows" text,
    "Insert pysical database calls" text,
    "Insert request times ms" bigint,
    "Insert  time/row ms" text,
    "Work process number" bigint,
    "Maximum memory roll KB" bigint,
    "Total allocated page memory KB" bigint,
    "Maximum extended memory in task KB" text,
    "Maximum extended memory in step KB" text,
    "Extended memory in use KB" text,
    "Privilege memory in use KB" text,
    "Work progress in privilege mode" text
);
    DROP TABLE public.sheet1_old;
       public         heap r       postgres    false            �          0    2815622 
   logs_fixed 
   TABLE DATA             COPY public.logs_fixed (started, server, program, work_process, "user", response_time_in_ms, time_in_work_process_ms, wait_time_ms, cpu_time_ms, db_request_time_ms, vmc_elapsed_ms, enqueue_time_ms, enqueues, program_load_time_ms, screen_load_time_ms, load_time, roll_ins, roll_outs, roll_in_time_ms, roll_out_time_ms, roll_wait_time_ms, number_of_roundtrips, direct_read_requests, direct_read_database_rows, direct_read_buffer_requests, direct_read_request_time_ms, direct_read_average_time_rows_ms, sequential_read_request, sequential_read_database_rows, sequential_read_buffer_request, read_pysical_database_calls, sequential_read_request_time_ms, sequential_read_average_time_row_ms, update_requests, update_database_rows, update_pysical_database_calls, update_request_time_ms, update__average_time_rows, delete_requests, delete_database_rows, delete_physical_database_calls, delete_request_time_ms, delete__average_time_row_ms, insert_requests, insert_database_rows, insert_pysical_database_calls, insert_request_times_ms, insert__time_row_ms, work_process_number, maximum_memory_roll_kb, total_allocated_page_memory_kb, maximum_extended_memory_in_task_kb, maximum_extended_memory_in_step_kb, extended_memory_in_use_kb, privilege_memory_in_use_kb, work_progress_in_privilege_mode) FROM stdin;
    public               postgres    false    220   <       �          0    2815616    sap_logs 
   TABLE DATA           }  COPY public.sap_logs ("Started", "Server", "Program", "Work Process", "User", "Response Time in ms", "Time In Work Process ms", "Wait time ms", "CPU time ms", "DB request time ms", "VMC elapsed ms", "Enqueue time ms", "Enqueues", "Program Load time ms", "Screen load time ms", "Load time", "Roll ins", "Roll outs", "Roll in time ms", "Roll out time ms", "Roll wait time ms", "Number of roundtrips", "Direct read requests", "Direct read Database rows", "Direct read buffer requests", "Direct read request time ms", "Direct read average time/rows ms", "Sequential read request", "Sequential read database rows", "Sequential read buffer request", "Read pysical database calls", "Sequential read request time ms", "Sequential read average time/row ms", "Update requests", "Update database rows", "Update pysical database calls", "Update request time ms", "Update  average time/rows", "Delete requests", "Delete database rows", "Delete physical database calls", "Delete request time ms", "Delete  average time/row ms", "Insert requests", "Insert database rows", "Insert pysical database calls", "Insert request times ms", "Insert  time/row ms", "Work process number", "Maximum memory roll KB", "Total allocated page memory KB", "Maximum extended memory in task KB", "Maximum extended memory in step KB", "Extended memory in use KB", "Privilege memory in use KB", "Work progress in privilege mode") FROM stdin;
    public               postgres    false    219   �       �          0    2815605    sheet1 
   TABLE DATA             COPY public.sheet1 (id, started, server, program, work_process, "user", response_time_in_ms, time_in_work_process_ms, wait_time_ms, cpu_time_ms, db_request_time_ms, vmc_elapsed_ms, enqueue_time_ms, enqueues, program_load_time_ms, screen_load_time_ms, load_time, roll_ins, roll_outs, roll_in_time_ms, roll_out_time_ms, roll_wait_time_ms, number_of_roundtrips, direct_read_requests, direct_read_database_rows, direct_read_buffer_requests, direct_read_request_time_ms, direct_read_average_time_rows_ms, sequential_read_request, sequential_read_database_rows, sequential_read_buffer_request, read_pysical_database_calls, sequential_read_request_time_ms, sequential_read_average_time_row_ms, update_requests, update_database_rows, update_pysical_database_calls, update_request_time_ms, update_average_time_rows, delete_requests, delete_database_rows, delete_physical_database_calls, delete_request_time_ms, delete_average_time_row_ms, insert_requests, insert_database_rows, insert_pysical_database_calls, insert_request_times_ms, insert_time_row_ms, work_process_number, maximum_memory_roll_kb, total_allocated_page_memory_kb, maximum_extended_memory_in_task_kb, maximum_extended_memory_in_step_kb, extended_memory_in_use_kb, privilege_memory_in_use_kb, work_progress_in_privilege_mode) FROM stdin;
    public               postgres    false    218   >�       �          0    2815589 
   sheet1_old 
   TABLE DATA             COPY public.sheet1_old ("Started", "Server", "Program", "Work Process", "User", "Response Time in ms", "Time In Work Process ms", "Wait time ms", "CPU time ms", "DB request time ms", "VMC elapsed ms", "Enqueue time ms", "Enqueues", "Program Load time ms", "Screen load time ms", "Load time", "Roll ins", "Roll outs", "Roll in time ms", "Roll out time ms", "Roll wait time ms", "Number of roundtrips", "Direct read requests", "Direct read Database rows", "Direct read buffer requests", "Direct read request time ms", "Direct read average time/rows ms", "Sequential read request", "Sequential read database rows", "Sequential read buffer request", "Read pysical database calls", "Sequential read request time ms", "Sequential read average time/row ms", "Update requests", "Update database rows", "Update pysical database calls", "Update request time ms", "Update  average time/rows", "Delete requests", "Delete database rows", "Delete physical database calls", "Delete request time ms", "Delete  average time/row ms", "Insert requests", "Insert database rows", "Insert pysical database calls", "Insert request times ms", "Insert  time/row ms", "Work process number", "Maximum memory roll KB", "Total allocated page memory KB", "Maximum extended memory in task KB", "Maximum extended memory in step KB", "Extended memory in use KB", "Privilege memory in use KB", "Work progress in privilege mode") FROM stdin;
    public               postgres    false    217   [�       �      x��}[o%�����/�<d\$�uitt�ㆻ�=�z�$��؆g� �>�"Y�E�Hn�X�4vk��"���u�����x�_���|��?�������>|z�������o_�|z����W����?�z��Ǐ�7:v�0�ߚ醱�ю���?�����w�FڛxSȭn❍��7�0�e�(���9��H�����?�Ά���&�7B�~�ֵ�Ǒ�Y�q����������GG��?#�ONԓ�wd0��'j�e�O(�T�\�cţHd�����35�xl&����M|�pk��jNn]C٭{6����3I�Q��v�KRv���v�`�k[��I�F��ydү��o�����Cz}�����?d����ķ0z8>�T����7�E�����n}�l�po��������;X�׊���<L.�T};F՛�Ѩ7�㚾��;i%��1��1M��0���wl�����'h��7���&g����q��(������/�ۈ#1�A�A�c�_q�M{&~�z|6�����4(�x//������i�[�)�7��;�!����pS��G�D<�|�O$~�o��čb�'��E0��(� ���#/NC�G�'"DA3�H�ʣ��✩�G���R^Pd�4_>��Hz���"0�������9��i�mY=�� �9�#�⦋_��yu$D�5�ӡ#���	�H�Pף��Q����]>�Gi��X��m{ym$�[-ވ����u�+%O�?I������c6R���c�a�>'&ODHl}�)�OD>G#O��q�9��9������Xk���>?��=~�����������6tZ�����wy��V��ѥD����W�mi'��e��<���R~�Jq'��_\?1/�\B���:�C���+Ow	�0����.��Ԕ�طӱ�ӱ��&�#�	e"��M�/yi$!Q�B�n�m���S�%>�0��S�	zH-;L�����X��:�<��y�~�t��0:.�x��C�׭ţjJ��s
���s,�ԅ���F��!Ԩ���]�9x��P_=G��W7�2��.:�w2_6��n��sN�R��/�T����R����j�~'�9�<�sܭDc�����1+aFH��m��'�N�1�vA~�_�hP�A�Y#��`����?Ǿq=� �y��.� -�Yo�nǠn�(�@q;��m��v��r��}7�C�Jn%=�,z�E �N���@A�7�zD^��QH3D���<hG��=�84�>�IH�F�k-�)��d,!?���&��]��ח��oo?�}zy����������_?=�f�!���M~c�[��n�LJx.�.t�JT~�1��BW4��V��]~l� �G(v�d�Ұ�7��gڙ�?1{/��=�ć�G:����G(�1 e��4�n0��Y�U?S}�D%J���B�	�BXSTrؒ��㴛��2����yo��_0>k�퍆u�QV@�x_� ;�J�J�[=���S%��¤���Ƥ�T0)�-$:!f�8�b�G�q��_��^>����˻D���v�Q��T2�M`mL�ĀVjF�e��64�zP����4�k�V��)D6�i����v ϴ�0����c�]`��h���ť��06��=�I�EԱ�	�w��)p�������7�D���������	�?���~� �?A�o�淯��~z���,���pC�ܽ�4#f����y���$#���݂�@���/�HՓS�R;�����h� ��
��o����x�X�ң�*�5�-<��@7�3������HsxP6�0ޤ�!0�� -��4��ɭ�g:�����|���5�%f�5f��(�'ٗ�1w����%��A^���++�pل� �7�������]`��;�������72�׀���'QL�f#f��߰H<1�8��I��	Ml"e��|l��:���I;4����s��,&�Ds�ϯ')��]�*��Q��E��JV������p���T[׌���̗ͯ͟N�h*�����l@Zeqø8X��,�Ҽ�?�fVz\������Gl �m��D���+�\;��4￾|������m�8���k�p���݌�&��-�$�By�N���-�״���q�������f�d;en���iy�!`z3�邨D	�(yCd\��|(#��1)������FzO�Z�>��X���m����;1�=H��l1�Q�[yC�Ɉ��yބJV\��.��0�5#]RK���|Hiq)'Z65 ���ȸ� �%�G�h^i���h��L}�[ �Ac��&$�n���d��Ri7�#�EC��7���J�����`}�+̃���&o����3$���M�N�P,��BYan��ޤ�5m�I%��V�u�M�m�얯̐Xpc��3�V�.�H8N��ST���'mc��S8B$vao�7�.�mkЪ���Crb�0������C

�M&Ӈ�
0o��d�s��ݹ}��,m>޻�BT���)�J%op���l��Q��oo��{#�]te"t�!r��Խ�	�`���������p�R'�C���ፃ�_  �m��9G��2��L<B�F��@v��;�d�W��R��/ip�	:��`K�]�k���0�t-��"�7��������h"����� |˸���Z�Gp2�̀�F'S���N��n�)_x�/0�`�A2�����F�@5S�*$�X��X�4��5�6J����3�k�+}�AR��%;��)�G6y�}��l!a�~}�ݣA�g =|�%C{[�Ҳc��`y*.�7�� ү����Q�^�jꍍė/��緗��������/ϓu'���[���2��h�|6YH1 �OG�o��AST�HY����%J��&��6��V�u-����i4#����ʊ,k	bf"L�?Y�`,m�����Vߨ3�q.+F�{:>��D��c��H��ܐ Z?D�<[�`
|sH�����G���(z���O�?>}zz{z{z�,�������u�>��F���ˠ	<��3���,�͚�ر��������Û4��J�Oe�`Cs���TM'Q��Fy�e���PDZH'�h&]���m�ʙ�>J��	��)�ERt U���.sR8I;��|T���d�
&dm�0Q��];L��@5;4�{��X�Afqd��� vv�2���ʀ���nKF�;'e�˥<D���oᣇ�7�G�Lf�h'�����7��+..��A1V@�A�N�J��B��*���Ӳ�W�*��:���\~�'/ߵz_��DV;�B����fj0�̊��B���4WL9YG��0Y����Ҥo�ç=�
�e*-=�T����ݘ�	 �z�Kv !e��P�&�����S�hy-�
(YP1�Q��Y����_aES���sTh_�y�!�7*LkD��rCAAwc�L�6J�����^ �����/P{�R���4����(ؓ�
7�`�9nJeq!��=x@S�%��@ֹӿ�U�"k�/�P]g -fi
�1�7�L���!+QW
D-�Y`m�P^7�3c�i��{�*@n�\�q~�������|����{�b^i�f�
=h�n������87�Uj܀�!�O�J,2�V�� ��=Ⱦ��p 6���Y!�5�|2��+X�m�c)DD�vK����1e3�JA��d�Z-+3�wT�[���u//l����]{�C� y�^�\�����Ua`B=�C�[l��Y,.�Ε�k3��������I�	F��J�kx��Sfw����>��Jt��v5��: Ual���27��ӟ��z���~{����>�
�[��ح��`u��'�l��Ҕ�cg�2�|�ׯ_~�U|2N%����p���,��Qm��n:|w�˨W��I�VX�D�ִ��S
� C���ʩr�Q�I:�2�O!DX��5�~    �MK�����/��˴���bM]6���sA|��IZ�`AWГ�?N�9U����⧶�&����[E����p��iG���̤��h����!����e`\���$�����v�-7ij@�#�1�w23�f��"����g�,@�/�3�f��"��Q|y�=Sb����\	��ᠥ��!�0]�>B���&i�����W�睃lSq��z�<�y�&u�N� tU�Cnv�s�<T���v��<l�Ф_a��sծA}, h}q�TA������ޘ���Ḹ|��1�qr���	nl�2F�0󜰇�&�'�-:V>'d=̇O�Y��Uy�ul�]K��"D��OB6��ӷ�o2@��XAiP�T��Vw\j2�J��̒����O�ã�h,D��J�����1�*mC���ܾ�JBa6��;Y��������FE-?g���
�*4C,�A��	���z�����8���/W��b�se��mҹ�l�2]���ߜ�ha	�G����L�~kYMN�W@�i�gY%�Y
2�9���I��А�r�3��� Zfd��z�	��B#k����t)0�U�S�~)�eF��*z+}.�>�B�ŝ�6��6�m���_�O״�8Y	m�A��
z[\�C�Y��Zf��!s��sӤF!��3Z��AK}ÔV��I��I������9�6�y}\��1�;�.��{�������<��Z�u�f�E�"�ٖ��<�;~-�����ȹn{��U���9��XY�DHk!sJ�4�#󜦴��5C��J���)lm�r+�h.�Ҧ�2�o��
��Ũ�s C�3?�[w�4��<����v�,��0�����3��b##���q�iyd�R�(��n%�A�P	�@�s�&dr�����ۅ��y�
�j��_�g�!�����Y�8wt��]�h~�֐sgH���j׵���y�C�� �ԀY��I#d��Z��Ȝ�ic��U�n޶�QsT��q�ȎUPZ���zedAj�����fG:��};���-@y��<Ϡ;�랄���!�7\U�f!Q��p{c_nC�#6���� ;6g�����D���XBŐ��@��-�,Q=���|[�,5�:�Z�g�N9j&r77��7��P9���G�J�+M�jW�[}Ջ�>�7��!�"�V���p����dj�xoAr�%[}���v�F�3�8����5��°6�dʜ�3&�#���K��x�Hʹwwnڛ���ls�e���<{�'���C��gC/��<{����Z��x���s[�3�nЇkpn�S@�E�T�G�Z��/&��>Y����YJ>$��T��Y�嚥���:^9̦n���3��e>� �`?���b{����6�ݣ���-�բ���6�9Y3=��T�P����8����Z�n�I\�d01_�Y�ɱ������|)���r����:�i2��La�e!)���F��
BS��ԇ��l� jߛy\7=�\����[4b���G
u]��M�U�Oڪ� �{�P�g �fM�^g����÷�/�O��^>����˗ǧ�׏�?�R�N40�%��FCٵ�z����o��#|�v9r�mc�L�K�e�Bj��|�d���5莆��Ta��W��F�q�F�B+���tq��P�0=��L#fF:���	�3A`�;U��0Ð�(7W'Ӣ\�l�"7��`BMn����(}�V:+j�.5�'��0���jqP�J�LZ�TM���3-�CӁ��6���i��b3��3s	'�:��
�Q�9���g�Cl�p#��]N6e��Ǩf�&c��d��C�b7J�3��T�-aBi��:�e_�͛J�;f�	�g3#BR��Jc��K�&�hj�:��R|!6-��h�˫�B��2:�j�6��RN���@"\+3��\���3����ٷQ���޺I�XV8�hF���s�4#��f����ҌB2�O�����߉���`�dB�߮C�0�n�dq�տx����.��ۃM.���~���|y���Q��0�Q��ﴷk�['���6+�i_<>~~]�l�s�d^r��r>�\b�8���o	+��ͨz�}�$*�h��a�i&G�98�x8�Ah��I?�Xp �۪��;R�����q��C�7A���ٽ�F�&�Ӳ�%6�	�	٦+��N&�d�mj���Uh���3�L��fa�r��L��3�Z�7$���}�����X��>V=��>�#�ڸ��V0�3����z��vK�<O��7�eD2�|�y���hn����&SshV���y���C�<s�@��73;�45W���ȍ���r{��-����Z�㑙���xd�TJ�]j:q=��*2�/�6�^��>Ziܐi]"�Η�dg/A���jQD������'����z,�Ζ�TK��߾��-�wC���0�{z��)�嚡�
�F�yFCP��ȜN��+�����,��O0oT2�qrJ8҅�ia�y��R���y�8��8��M�o�0Ŧ�Y ��B�z���j���r;����$8����uΆ��x!�Vy}G2'vʁ�U��r�c?��nf榫�f��<˅��u�a��=��� �A�1TC"�L�e��l$�g΄�f�l΅mj{�׻�3�8���QQ+�Z�M�=�m��G�{�K�2#5��o߁�P��i�O�o�� m���,��$591�E'u��K����w���b�N?�I��qk+ͫz�	�y�3)F�;��r͚g:������J�<��nUdj��igǅ�Եi7��V�,��]fy�:��)�D��F�R�V[�v^~i#I4쒴]#E]�d�D��"L��]o���gU>������3%�ǔj�-�~�m�v]�;��܆R����[��q�o��1�	_b�����4B++���HJ椤@�*�,�|CI��l!�m߳�,in�}����z`��1�p���hs��^�p�h���w�ڵ�1�o~�h�����&��7`������"1 �y�?N5Az{P�#�E��(n�	ΐ��-lj����vW.M��4���U�I��!�l�O�Ju��踕�d�Yiys�VuG��ق$&d�D@��;H̅P����;��nt@�I*&A��]��G�,���xH`�<DwX��9 ���@^���!�����戡;�u��Ƽ��_R��z��h��$����&������)���xyk�H<�8�����gP���ڱ���`��ij���_�!�*x�BH�k
G�/��P0��ڙ	� �Y��ˑQ�je������n�����5�J9�w3P_�`.�0@i@�`/s7l-T���7��X�E�j0H��A�z<����6|9Pp��b	q�^AdP���:�NS��~$�W{��^@Է����1Ɨ�&�[�T��.(Jg��&/���vpi��F�A�,��O;�����X��E-���8*�*^D���w�AdDSbTަ!���M�I�a��%�w�Kf��>;���6���9��H�c@:�tw6���^�q1�~��#�<D�v}���~��ߚ~.�"j)%��n�ժ<�'F��F�No���.FQ�V�#tkKlYa
��F0%��R��[T���h�T	�4��av��o>8;�K���0��qa��G\�����<D���}���G'/���
�7^�>���T�q/�:�a�X�����\<e���m�S��@v�޸r��$�c�R�d�? k/�r�d8�����Q8c:�f�3�'0F��8u�MT�.mT���Yk���u���v��(M1��� �~��T�Ҙ����<S���)�i�I����6S`Ҫ-��p�Ɨ��ʝ��z*ۛ���Y�� a�I���\�*�жt���f{3׌ț
m�q�2/��磾��::������F��-���p?�m�o�ea{�KŖ|�����^2�o�_�1���*��HOr��cғ~3�\�G=9�^2���l��ߒ1��`�j�j<w�`ㄘ�h|Ttz�X�E��)/�;]ş�X�Ǝ�}�Ik�����F�    ᬨ^9��oⰮP��K�V�|X�Q�6�K6��UK���N����s��D�vQ6�a2�1�:� ���ܝ=T̯Z���PЧ��s+p�EJ�{b$��C�X��!�P���۹r:�"gu��;w��wJ~��*vo�3��]C�t�ҝ���yef~M�`�{<5O7�	�RM��R�I�Fղ{��Ih���k1�`L�w���ߜ���8PO
/`zȌ����G�8OS�4��Y�I���:�t�:�;��,�(VOg�͢j��T�����̅p+d��vc�J����*�)��xly@H�c��uB�3t�N�^
y׺G�W������bv�OEg�iy��'��k��7�r��=G��@��53&��F���b1� ݹ�;��?�:o���%B�^B:���A�ϏnW����.�ʹm����ޠ�S6�؃���[��ݗR��.j!��`�Q�ץ+�������q��#��`�k������%:�a��I}d�������N��1}R�8����]I���IHt*c���o�q�K:S�3�m��=W�۬捌18��C��ڌ��y'��f;�{�����o�'ֆO���� � ���f��@���vC��esE׵(4�=���jQ<�L��ڒjAq��m/B�6I��
o��S��;x�mM�}h;���C��P�ԩ6Z�z���\8Z��� �β[�5�\�)sH�*$�Vb@c�⡁?ȷ����q��P�s�A�(�6u��������'���p���A��=�P1&N�_1�����bJ��^��}7:/zo�sl�<v*��э"Ă�^IVy���B��[.C�I|	xn��X���)aWG\1��ٿ������͍۱�"~ 4�����������h�5fh؛k ���5\�`
j��NP�yC���@��5��M�;Ȣ�����"�!o H��~�`d�H����!d�{Qb�znd�8|�	�J��0��e�h9���e���t�����ni >5�<�2x��+�Vh��t��4m���R+4]&��t�?#`�f��ҕ�Qշ�}}}z����Ǉ�_?�?~zzx��u�c�����JTj|�.��E7��Y�lVui,~ M�A�/|�<�v�	�랡y26���H�uצ*������� ����d���)�JdU����f9����RQ!�Y��v-��i�#>ik݂��������?�������\�>�\�\����V����ѣDmӵ���"�qTo�eU,O6I7�ւ�>~&���P�����Q�?���ˣV��\FE&nf��Տ%�?���Ϯ��X֩�,����î8��'�,�5�3Ffh��e��MM7�	Ꝧ��r���@�h����L˂����Ϗ|z~{y����$>���F��!���i��۹�UWlN��� K4�|b%�:uc�^Z^Z�m׮�\6���v�4�yA6\[l^I��z�s�6QX�G��}`럱1�>������m�h{�����3��\�X�}󙚵Q���%�9�1
Zl����Ò�M5�m^fK��f6��kS����}X�����M��bզ������Rh`�ĥ�4�]
��&[�U�#�Y�y�(�:a�}yl��a��fC�XA���tD��V����h�c���L���O�u糤C�_O�6��ɮ_�V�b�t����'ʧԍ�D�=��064�#@�x�Ɯ�`��m���=���^R4]�M�F\��[J�-5�>�?�~���3�drf���[ҫb�`D��~�{Y��F��%[�2�e���͔���?05��sa�.�oΆ�m!�NTC�3��HCX�zN��m���Ot�Z��l���,�#Z���
]dR����ۗק�o/�~����������~���ܠm-h�	�ǁ5ov����v�{
ZZ7�\Ah��R$������������/�����������?������o�����ʛ���7�_~��@ϝ������d.��o��Y��a��������r�xkuv��o�+;����$flDCi��D�	�;�b#i�B��2�#)r�ײ�I=��V�̦��mH�H�;�Љ���e��Ĥo�g�z)��0��ã���beu�J<f�Ʌ6's���?�q[U�6{/���,[U���}�im���w�(�;�
�x�\g5Si*����(L�ޡ��j���7+j��h�
�j�]�hi5Zh�܆���uD�vu�p�='�&�X]��&~��"~����>�����Ia0�G-MnfۚhkG��V���%�r�JEkĂeI3��P��V%ܢ��	bpb���m!�.���������
Aw_�)�%e$���0�F\"��6�8Y>�+fU��A�K�	�����������Z6>Ǻ���X�d~+;
@׳���'�6���&/v�mLdZ[!�#�̟��6�z��`�;�(OU�*�l.':�������)�W�����q�ɶ*;��2�5�4�ML>X8��#��|�[rEC@���r Չ�� �f��B_�U�JԔ� �G܂OA?#Ei�*�]y������a\Gb�1^��@y�M�+�@��KOT��
|� '�sj=vd��}j*@sK�iF>�"G��C#M��R?��<�3o��-�4+��߈�f_�����۶�1$b�@/��F<���o�!G�K��׃�������{�w1�Q�̖�8���s^��e����Q�e��|l�҃�R����ȳ�=�jm;�6�gW�~��C���i���ԝbA���5���.i��dwł<�������Uw�aF���=WOժ@�~��mӼ�\�U�n��\�[a��Jy��4����	�':4����y��F:]y��ɏw��C��艌��l��X����z���Z�s�8}G�7�����m�����}�вK�K�|��] 󺒥�ܓ�����:��lJJ����vBP��|��>4+m[/�,�o����W��&'�ezŠ{���d�~� /P����pޥ�%@νГ}䅼.�[K�&����f_B��}��C�h��u)�2	�fSYh�ߨ��KUGH��aբ%4���I�V��hy�zbERw�F�$�����b=G|��1Y�U��Я��m�'��Z�][���ƔS@O�*`��{L��rLO������=�4�����������D�-NP���&X�۴<�nj$��%cu��\���]�Ss�'���0/����6;��S	:�qp�09Eta���饡_5���5]z{]�;�:��Vz����j�S�۪��Ǘ��!��B�o9�wW�ꗆ~YC����/Q	�Nso�V9	����E�+t�=s׆|�Q��.�jD������^� ~!G�K|+�F��|����YF�����-d�aZ�E��ܶ~!,1g1x�r��9q����T�]��rwpl����c�Y�����ի.=��B���~�+`l��R��6�V�1�k�*CW�zӨN�/{�z� �w��^�ݳ�+A'ɼ~�ץ��ݣ5�� ݾ\֤9"+t���"(��*>�y�k��(lC�FG6�BK��Z�tgvuj���MBz ������C��ݽ�x��m�sW[���Xe����%3r��ȓPw���o&l���a$z/@��5�֡�U��<�6l��	����D��aI�\-6��yl��x���u�o���xjw�	(>��<=�Ӏ�Ծ����l�yGӓ��� yj���/*��G���KN+l[��T�8��
� ���v4r4s\/�ZDp�z}�'�����eY8�I�@�Lm�����kwS)�N�A֧�Fh����ۗק�o/�~����������~y��	\�[���z&����'�pG��-]�Ld�m�� yys,"+���������s���<Wlu������B��}��@eݱ�D3�L�k��U�B�LW���U��p_U&�ؼX�55I�s���g�����䲏�$O^0z���    �^«)̾��)�i_�-%t]z��M_Ց�l�[L��I!�>�>=����3bN����re�M!J����OQ2A��T�8��^O~P��jo�9�e*�i�9���NP�u����Z�|�9���B~UnIO��G���<��rKjFt;Р r�D�y�D� �	m1���!O_}q䗥�EKBH�H����yj#��y�� �(���&��ȓ[^k������u_��8�k�'�*?Kq"Mn��N�K#�vuGv�&Kk��&��V�hr>�͏�3'W On�<�l4�>�,��l�+\6�.�M2�*{�mLo���<~�쉐_�[���VG:6yU,ׯ���=����q��M�Ϋ��V4穀���D�RLWF���m�g.OnG�nY�?F���8sy(�߮R�NGaj̴��q����X1��Kj��/��㇖\Sܥ�}�V1�E���7��eiίiv3rYnI�Sz������Ƚ����Xr4rk�E7/h�e$V#�
6�09'�$SʽCE�Nk-����Z�<4V)�op��d�Uj�l�U�����/�U�C��� OL���E����'v�Ԗ��C�I�^V��i�Wa�x��!�E�	���^���m���.���ƣ}�>�u�'P���7Kۀj��Y��~�8BYӗ�s������^yz�l�!�,�]�l�-�ߵ�~��Hr��wv�K1% 7��V�����\R<Q�Xpg�7~��=��s�:�E�V'�G�ݜ�.[d��g�~G~r--|�Y(��|��$�~A�T���S7��F>~ i=գ<�m��iwY�k������������)��DI8�9�Q^������v����
�t�Ϩ.�dPk8��QC�@�-r��vd�#C%�D�W�u�-ݝp��U[��Ű�u���<� gi
��s��>?��ύ\1�C�<T��� �Z����řj�Y��k�qׁ+y�w�jk���u�y�A|�s	��K���c�a`���11�U�����Op�#��#OFUyb�=Z���w?{�o?{�{���#�2�ȯ��h��W^ܒ �D$�ffH O-Յ�E�sK�������M|���^;mAmG��bI�Ip�j��a����A��/��3t�_t4��|B���&gp
r*`�@�祺Ürj̖K'`���M��G��7gM���gC�g�a1ل|���DЭ����`C�<�-3��A��[O�o�?��v�S@O䗡4���[O�"��V�ҩYr����LYHe������?�z�ɱ���(�e��.k��=>~~]ͤk�_��o���A�d�	��[��҂$&�mR)���%�Pc+w�LW��7�wKL�i9��怓tM��9JbP�.~�]k��s�rT�^;Bϔ��i
�\�ʱop�T�U|9��4奻��{�<��u�^�S��,'3�/4�q��I�dL�;�֜&~ ���MEy�I���*x�^)�ۗ�o�G��8Z�kLSN�O:���M��4^�m6 2[{d�1��缨P������}���sI�nsm���� 8���th+�	���N�lAB�}{���.풖��n+t�>���u�����u�����e��A-.��!h0�'�F���I�ak5Y϶kʞmg�m|�˩�a�ܡ<��$w�$:���s]l����9n<�&`�X!���ނ�<��:��$� ̕a92��Sk6�Z ��[ H��_w[ Z5bA�F��v֍9�*W'F�0'���g���g��=�,9���89xr�6�[L�����$G��>�l�\ok��u�"�@O#�/)h���6��}�-ܩ?XhIp�L_a���1E񨋘e	?��������N#�l���x�6(�VEt01Yj��o�Eف�h�l����a	Ș	7O��]7�@�)f���^{c����/޲����C��%l���'(�����w��19��ھ��䶖�se�S�!ǔ���r�9�vj�b��R]����h��z�r)磋���u
2������Al��ɸ��������k߷�I�=��'5F�8��PeW<�7��I��V98͒.6G2ߝ�c	�Ņ�o�d 757�@Ԅ��FLh�͓C
Y��&�XXb��&�7b��B�,�F�jk�\����N�e�s�7l��I5v�B�*���~�)����K��l��	�DB�麨vU2�� L�q�69I�PL��4�B~w����3�Ր�m�89}����w���ōg��RF�E2������Z)���p������4�Y��e$����w��6ߥ�l��K�����*���o�P5U�ݕ��~���������J̒�w��'�%�xYɂ���In`����l�C5'�����b����k"�����;77����S���rJ�h;�r.
�$�W�qD$���%� ��U) P� R�tqu(���ĜK��x��U�rN3tV!yr8//�=5�0��;J! Tڪiw1��U� ���/*3������P�YrThֈ�C�n��	v��w����8�h�V׼���~䩆F}�}�ߑ���O�sِ�_���=�U8�c39�t�ȟ��;x��������8�i���$g�c�`�`faŹ��0�۴�������{x%Ӑ�
i�L�R�#?9����ȽM9K3Yl�7?!B�^Ue	\J���,��gWN��wŁ�Q��P��~\\�'�Nj��Ƚ]KM��ږ��w�}������{��	dglf�X`�ඍ"wU���@�֗c���=�f�/��S��7�u�����PLi������k`��}�����^r%.>�"Z����׷4���ȱ�2�j�\��}��܉Qd��M/��n���w��g�EJ�lw�����*7����U�s0$�%�Kp�J�'�̕����v�]f]��'l��䮂��9s�t��e���qJw�ٝk��L�M/��\j��*"}u�UQ���)W�!7���L��7���GSv�B������S��N�����V��y|db%J}�^T�x���ф�H~.ߖ�v�X1�5.�/K�>�(�ֆ�ݺ���Z<�!���;^��]�߽]������#�д\��D��"�n�6���*��*�W�{hA�V;���t���aw��/���oX~�+�o�x�k]yU9�HP4��Wrȿr⇠���I�u�l���IJ�l�ʱ�B��&��VǝkHƱ�K��T&xx��p�CJ^�[9�C�W��~��k�=�U���y���_1�M�kt������,Dk�';�%�m��b'i\;�MMÜ��i�k��{�p'���}�[�k�bn�{�+49/S9���â�.��#mQ�Y=g]Ԍ�����ӗ{����mj��W�v$�|��ŝ<���L�8��g�}dl��/�KpT��S����Fv����A�������۠�%�<g�6p4��">KzF�.����^w✚��/Yǚl+�r|�%MU���l�
���W���R�jZY�:޲Q���f�aX�����U9�!g�᧕q�1����qm�v?�?��O�X�i����&���'Е�'OX�|��ǔ�q�gCi�l(C�ʊ�M���w�W�~~4�K�����WِZ^��o%����[�+�Ѱl5��id�2*[�Nn_�����m>}'�*仝|��J�7�X oC���8��k	Qc��� ���t���u3��O.�L���#u#��)�ʁ�d�h�{v�ޑ��>��Y����%�"�6�pP���_�ݐ���Ũ��Q��5�+ꄩ�}J����U�t���v���잼�O]g������VC,�[�8�y��xQes���h�h��m>��"R�S#>OM�N�Sc��[���Փu�~��ت:l�E��)f�}X7�u>$KK?W?���x��Ǐ�ׇ�Խ����:�=������+S��C�%5`����]T�a��C�I�7ԭ`�m#�]ו���:#l�a|Z��Xp�VtȰ��CI	!���8k���+�n:���#2u��v�R�    Ȱo�~�LR��~m��Se�r.���DϿQK�Ԕ�<������02����s�^0ȫ�/��9�6p�u��A����:�[���).qo����?<��<<���÷_����˨J׋z]m���?y�dh��PV�[�j7�����fC�埌ys#���G�֡@���v�?�����eH��bː�}�!��� �s��������_���[N���� �y��
��9�gP9��,a��ĥ'\�6tf]gE�q�d�D�~`ww�b�[(��	'@޷��������OO/����Ăc���[�:p�_�}�Nv%�0��̸mSj��q�6�����.�7�ڸ"k{eMK[8<�\Ѭ,ڏ�[�����ۅ��wn'O"��9�/�ws5K��Q^7�uuI�{;�׸�e�;�;�x��0�?==<���¶�Fp�-���ݠ�Jس�	���DA�q���pK�N���Ibߗ��n�TU���-���k���5�:� �gu�M�c��P�fpT�M­rN�Z!��, �yd2����kYnk	�8M�X��o�X! ܰ�4Pa�)��r��!��(bL�,�1��fI�/��?��Y�D�Jx\����t/|F��(y�]et��n1��v,t�h��!k��A�J�/̕�͕�P����HϮ>dfΧ�V)��tђ���	
����$<q����[r��"2cE���n���^ؙ���<X���,�0B7F k<�B�Y�4��E�,�Jʀt��P�/���.ݔLo;��R�8tBmC��m�B��	��rn�a��"�1�����sx �[�YZ��:� �m�e��v�&E1�w�#=i+h�Q�z&C�m���ՓA�ڲ��a9���+|=��l<M�d��
)Cт�ރ����ٖny;���\s�(���9@�yw�ϻ�w���&�_����/h׿��:6�b�Y�.xf��q��7�$)��K�͇���A=�����NB�@O��'5|��PS^>�{���#?�ߙS;�	D9��t�&�f�-Jlu�%Y����[����8�Ւ���2pq�'j[B����w���������Ͽ�㟖�/eD����[������Û��j���.h��J���ɂ�����ɬ;��E�Tw��B����~����Q�6˟	�B����-��.�̭�� "�$<����t"�6��U�N3ho��8���vc���a3���<�E����[�����(ŭ��7��9]_�<���;���B�%�a������û�J�R_S^�U�,xy����\~nO�������mc�Fl%��������ڵ����}d�S6V��I6B�t�PƝ�bQ���l �A9����|�o2�H�PSzY�_q@�"2'�F����o�ʍ����o�����vJ�*�{��û�Ii���m1� �ϙ�y��\�ԫ�5���*-�T���ք�!M�={������w�������Ղ��"W�g�cv�{"dN�� �X`�H��ۺ�>�e��np5�h#iƲzwϡ$�d�Ƈ<ƥĬ���5QP��뢦<��j�0E�(��#_�87���t2�^u]��ta��g�*�����V;���	ۺ0�O�ө��0�;/nL�S�N��	�{UMِ(O�&�8�S���M?��j�:��P�
�A� ��쬶��p�ӹ*�H��V���w6�������q�sˉ_t^d�F]�)Yf���أ��F��N�?b"ih��B����M��'w�!�������U��R� �t�f��)2�O��P4�R�L��F=�7i���1e�>Di� 7w���z�4�pMS��x���f��a�~���re.�܈!�۪��L��v�7d;`����9�O��yiu�)�8��R��p��Z��we�Ιw�G��VS �v��|I<{�~�5t��Y�:*l�7�t�ŹkG��w!����AX#�`E�9e�y'I�".s8&w3Mk?�l�4��Z��:w�)�<n7-���[��9V� Ȉ 煥"w���Zd�̥�.�5vk�ZSMR��g'gL;J�;�ܪaʬ�=��j�+�t����=�=o��8�����셖��8-1�S����-7h��i��/�<	��_��W*�p�ٚ*�*>)qu��0��OJ��4T�I�����uZP��%�"iݓQ+ȩ#�׃�ٜ�M_���j��;|8J⳶0U�o�!����c���{��U�������v�.%���d�S�Q�k����T-܇�L����xNC}vy�1W��ct�kB��&�PQ�����j^st�tm�^*Xv9��V"|��J'��G�6�/���x��cjQ�#orS�7�<h�Ca|��A���S'�ݢ���;	�T��8p/����&a�%!�3�0���/7c�NܬN�qT��Em';T]L�/Cս	�qҩ+woQA���Lй
j�ؒ����-�`�ZXg&I�j��\��6#.�Y���4�V�p���o�����t\����K\�J+\]0.���[��a�2�rc���`3Ť���x`xSǞ2������O�o\��˻�b��4^���Ӄ�������XP��&�y%�c �m�H؜� R���8�,߾��G�E	7'J��w1F� ��x��ޱ(�ވ����,$�k}b�����Qx�P/�x#47�;T�X�4�(v����\�zA��&2��H��R���?��c��>��LG�~=��˝j��;��xQ���<k;����X�J6Gm��U9��>,�T����5vOm>�eU�/����^ʤ� �K�4�}���S:h�]AC��/�Q���[�W�_<8�|���O~�.�}s��՗�2l\�.�y9�nl領�ݸH9��I;^���8@X�x���o{<3�&�O���J��Qx�����+�kb}�E�W�0������{��%���X�6<�����7"���d�S�n��`�J��T��;��ԁ��q�\�9'��F��S�bL�ܑX.��xԇ{_ �X�wl�,׿�2�֤4u�������6���SMlsr��g?��G���6���?�0��>����6�n�xo���T�/�U�N1��;�5���d�4�:��A'm ;���<>��xs���^��ׂ����%�l�<��1�H0r�^�+�M-�87\�rT�P8��(�V�R�0�I�Ko�+�����#�Ȥ��Lr��#��N*�^*FI�
JkE~I���T��+�	��Cyֳý���p�W�j�jW�Rʷ��d��k1��,ya��;MZT�0M�Ks�Ýk��W7Y�wTDǯ���[\�F��{pW�ki!{�	G�Z��1�ȵCc���p�t/�;�i�bw��0v�D��N����p��ƌP�ȳ��)�S�J���ߟwv�i�XE��N�K	�>h#�v<[�釿������m\�NNwߢ�$�\kÍ��.<��f�pm�*Ɉ�T�0����ճL��%��s��\=1�Q�Y�9u�%o��7޶>�8=_�b��pc��������	ڊrv�FS'�0�x{�~u���w���2�Gc;��3�-'<�S�&��_��W��4���lp�ʳqc֪S�_�	���y��:K���!s��[�K�
���]3�˯e�`���m;QH�ɿc7yd]�V��m	�o��&H4v�W]
FWօ;ϥ>��mQT�gOF�jآ�uzhhnrz�^n\Ya}�Qw�lp-~?/��-7����s���؂���;��n�T½PR_���&��T􌡺j�zp�5�]�s箥�6d�i}�覊ᚘ�4؀��ܳýR����v��5�����'XM�#���굮�YU=*�>���{����|5�W�Bbk���a��,@��TkS�6%����O>r5A鸉�	��Q���yn\�g;?7��7�Ʈq�7����4|4vPh��i���p�#-��q��u�##+_����(��q�r���<�
�E���j�\�N�Q-Ǌ]k�$��I[�X.*@<�����Ynd	�v%in���� �?��̢���M���W���F�����b�����F�Ҟ���+�U#*����7��E4�w�&���U��9R���f�#���%�ʢkMkߕ���|�^ �  Y�����T�R
%.Z3��g�<�ݬ>K��yF2at���01�Z��a��Tf�9<or�`?�a��YM$D��&��/6	տ�E��.$�\4(tOK���3o���w��6�����+o�����9k��@����x̞y��ǞT�j�됳c�%=�"���n�yP>+<f��O|;��D3m�N�Av���F��ýR3���=�SV�_�����A�\d$��$�*��p�½�,;�:��Z-�=��dT=���pd��e�4���S5�g�2\4�������F�=V��5!^��Q�A���(��o�xzyyx���o�����/�T�F]���e[�X�~��EJ�&�74y=��]�pք��m�3Y����h^.��Hӥ6�+	��=��u�����%�(��nK��^�׽�m��^�(;OqےU���K��HqX:�o�:,c{&uWx�Ԅ�r�,�#�+}؝��#�T�%Oii�tnT~�:ܸ
��pc�=>mE�飮9}敮�@�^�T��IV!���CC��+|��٥���ց\�*W0ܳTٵ��@*���E 5�r��@��o����Wi��a��Ҳ��zi�!�0.�m8������M<]!��.�QN�q�~��p��F�
���$���S -]B��H�=`�Z示�:�h
�8���e-�����(p,�z�r	񟶠�&��
��Р�moS����#�3�/�m��,ק�9�6�	�����@�Ǝ���%E!����8�q��O��,��F6~��s
��%h���&$q]���u������\r��M��I�8��]��`=a�W�~�I?�.VM���3���lq���m5)�`�'WXG4�3�+���X���1/�u��dP��7�z'k�
�.`��֏����d!��]��.��¢8b�Ϲ����>Y��Yd|��za�s��A|"Y���v�q��N|�h�ޅZg� ~���S��e6�f{ƙ���{� Fw��!�z'���oE�D C纍��>Wg�[<�+=v�`��aW�\�óE��1+l;4�S��
��T	\6U��� ��>]w������s_[��<L��p�������:��xk���-'�=eA���,�_��^g�pD��bb��� ��#�}���]̋븬�����3Cu'��[�ӹ$n�6�K�!�;H��c�������?��ū      �      x��}Yo$ɑ�s�W��P;
��(�lvwAuɚ^4�լ�]I�FX̿_?�2�#<�#U�b2Y����]nbo	}�������������?�6��������^�������߿{�,�z���K�/�����:�����������$�����o�������ѥ�/��WL�5���_I?�j�c�����o�_;�0�2��r�/-� �=���v�ז�A�f����p*t��y��wm1�|��/P2[�bBݵE����k��/B��A>g����s�����8��p!ׁHִ������/T|���3ti�&����ȵ��h���?AW���7��$�#ȡm+�`�G�3��'^��LB�{�
�-X�������w��S}y���ۗ?�7�$�%W��g.�����].���W��t݅��S�qF#���\ۖb��V��߀?�V�R���o!�-����J��Iz����u���8���i�N�����0�I��p��IS��	L���L8��D�O�ی?Y���
�v����
)ي4,�[Ic�K��讄cП�I#�S�õ��I>5!��w��IZ;{j�<5~������¿j�zj�|j����ߪ!�(�9X~dbA�k����_�=��N�݉6���O�Ii�H�~l\�4�c�x^�?F�cmx�D1��0��J>�m����(���C4��ז6�$Mh�F��i��)�(?S��a�H�oEZ?��B�A�����~l�i�.�Ќ6���$iÊ6��4m�"��v��	�c�L�O�)��H��VDh�mEҡ-�dI�=�I��i�､�X�c��(��M����|lV�Ȋ��R�	>��%|�������O��}y}x~}z����:n�;���+%�)�"���q�0��-7B�#��Szŭ���+A���A��NpG��O�{Kc�r����,���V�
x�`�����;��K�y��J5��7$º�����	��	��ʱ_����ߔ(+��E�� R��r�"1]<AN/�PfA�|����t��(��|
=��4�҄kЛO��Ҁ<�K6#i�Z��t��q��Թ�*Ikie�`�̀+Ҹ��B?rZ�/�?��ۉ��|�V��h�8!a��'I����ܑр��%���ZC�YI��r�6.�'��F�(+K�H�O͐�	wf����&#��iX}j.àu�0z&h��F��'B�¸��̷xfX�a�;���0���L��0�aDo>�g�=�Ʉ���Wm�H�����M74��T8QoX�I�:�0�	���8Q=3���d�a����	�?vz`Q��B�;�6�,���?��m�(Tbc>����ߋ04�����0-��-]��ׯ_^߽�������������~}����7���2'I��o/ޒ�mk�wB�HR��=7�<�e�������v����Ɔ5�<��D�����?�`��[��b�A��̪WMLGt̮t<1���b�A~b:�s<|~����Q��	w�{�j�����W?]}tW����XD�o*ĄG<�o�.�(g���ɠ/@��;�q�Ҏ����"�D{Q��V�JWI�{�w�3�	5��pڄa��QZ��AO/$�2"qR��D�J�*OS�`N��ߞ��>���ǧ篂������*�å��W��1�,C�0�D����T�.�k�hҕ�h��U4��wq*�����'`RQ��R��[
!2&̜Щτ��U�2HXd��*���k��\ks��{{-��:�h��M?�3���b���%$��V脷"<"U���S�*,mPU�[	��/>>�����#�2j(���_e�I],\hC��`*t�8#"O"�0w�� �܈@	��Q�w���JIk�տ�:^=�����%?<^8Z�_G����N?�^�`I�L�I.���/'"���t��ʔ��<\D�'3���Hڨ~DH� �r���_;Î�X?3��:N4�nh�gɳ�i���ih���k� �<�s��p�����]/^1^Р#o����$Ov6�ĆfnAE5����W�U(&���R���$�U�i)Q"�,��(��B:�7�T�d��e*l��L~�7��ՇM�C:�L[y@�+�l��&B�;��vH�#ڕ�(O�6B)%��[�B��X06fL`�Ώ�R=l��ϑ��:૸0��k'�=�o������L�ͯ�k�̍�`A�x�Nsyt���0Hs���R�9nn�<��|~����ÇQF)��c�RB�Q�Wr1�����v�k7�.GQ(�k�(�5q����i�ʭ�iHm>j=�@0�q%ԃL�_x�q��j�*C"G�
�;LY"�n�o��������]�g>?�k����"רl~�~:���6�ҵ�"�?�*�7�7i$x�D�/�$�k�Q|�W,2������ڴ�@o�Pf�X4N�2#���L���A�eF3	5�?�W �O�^m��&M޾��(zM��٥�kJG�A+�p�iEO	g#�c���E���AR�aM@o�a�yG* �hM���Z�h���VM��A@ű�h�]����Q]�7�l�	0����W�ܺ|F���r �둔 �X�aDJk��B��GW�2E\:�5<͍�yA�z�S]�gk�o[�uF��$o�/"'!x�$�vH;�=?Fᑮ�pM��\_�{�_- @8��sw���^
���^�^n�����?d�^qx�1�Ӷ#���U)/�ΊA�?��Z�(�����$|N�6�L�~A�|7k�ݨz�~�|� |�"��2e&��Z%\$Zy�,��*W%��M�E�p,ŋE��Kw��XjFD�ߘ`����e����"O��#�19���~*���8�!��+f��?�L� C+q�:����K?�k2$��2��>QW��kL�!�$�DŃ�ы+-����.����`���Ӊ�A*zy�E4-Ĩɼ�?��t�(3i���|��4�.��\�U�`�c�I�h
������oE�X��:�!<��L��n)�ǣSAO��A�]�2���%�h�[�t��]T��: �U,��^>�>���o�~|x}����
��U2^�f�+ﴗ�,����</"�VOK�RT
��e�i�g�Y/��w���toT�EcD�3���� W���a���>K/G&O��ۂ�K+p  r���ee�	俑qvgȅQ�x�D�����J6o
�$<�I%��� ��#phd�fЧ��n��'��R�Sp'����_�z��������������ǧ�){�;��"Z���'�� 6DZ�1�6��m�]����痗ׇW�N�.Bau��H�5'�^��k�^��7�P�~	�lQ}�$�߂�p�@�ŵ�wV4�),y%��+ۿP3S�C�j	�'��7�!���;�7t��~	��n���a�&�Ӧ�H�|�Z`��I��#���G���~�i��a@`�K��� ���5#�P|Hju&r��L3T~�2�Ngo�T�J%N*F����I'�i�BQ�k3��L�IZ��q��NگBc�[^��K�ә�Qo{m����/���W,�<���p�%��|�D~(g�U�}A�Q�0o�(PҲ�$�1�4����^H�Hɨ��	��/W*"b�!�,V�Q��JiU�+鴊��Fe����Bp?�����6B�I'�2B�����k*kP��}��c��}+s;⬾i��e���#n�03���\�W�H(
yש0vT�����j�԰Em@����ҧ�H{.��|��,
�g��L1��v`���!�:�+p�˔�Ƣn��a;If˻_������=Q��c</&��-����х0\�Q�Jx�Bg�l�����ԉ�ڜMr2YS�o$��DU�L�vZ�w�����uS�����������a���/�F���N!.���k�7�n̋b�_����zP�'q�eE"ߋ�5��r�GH��qne���"��)��R��Jc(��R�U�p^�3	�JN��S��>�U^$~K�v��҉߻��Zz�E�OnD-�@�l�'q    ��q�����=(m1����O��dKO�d��%N��}�ݦ�W�t���.0Ѕ
�p4��	c&3�X]e\3����C&=e�Fϊ��k�A�v�����?������������|��3���_���u<��S�Ⓘ�b����$c��p��wh��y��`���?�7��X��8f����lv�{���a�cS��A:(.��]�!�J���]G
��J���a�$��H��iW�����l�p�tX��U��yډP7��%K��/��>~���D
W�R[�f�/����BV.�"&?��[%P��R�EE��|iދ(�t]Gb���Fƣ�V�M��6�G*�9ʴ�Xj���A�	{��
V�L���X��{�)�q�lW�cER#nB�d�M������L����8\H2�����c�u��:�C3p0��`�����w��t�m�O�=��þ����v�
��\f#NyYY�l���α}�Y��Fzة�}�����.Sr�dg�c�G��5ʢ�zP��ɮ�
yRv��C���mծd����	� }��:���&��qS�g��$��B@|B����N�L����+�6!�WXy�n+\��������//�����o����2?ի���73ɱ'_��Ѻ48=��u0�&��R�?K�~�#�R_�������!���`U�r�Ws�	����Vx3�ʟ`U��-Ҡ"�5~Y�!:�.Dd�|�;q�K�0�׿�4��nle����ק�3����ȊԔ�c�n�N	ΘOC�ce.}䍚y�ݬ���7TSL�e�>5%�`](G��.l�r�ا���RGj���Ǯ����#�Ti�D��kwû�LJqW7������[�^���"�%�w�����	�,��D��4i��F����=�&���o��2��7D���)*od�_ĂW�#0f��t��q�aW���|Y���S��Ώ^��YD6
_fG�`���YL>� �ɉ���`ĪW�8im�fR*�42�ڏڞ�ڞ����ʋ�gb���@r��MzT�����6ߗ�����E��7v�h�PLg�b6��Ө;1w�=�B&PG�ģE��s/i�u�k�� o�����z �Nx�%A���>���֜U�urҩ�m ��!�.��f �q2hDDC�Lq5�	NwI�!�=�����Y��Q�t��
���k�	d1�'��pғ��G��&�,� �[��n�sc���q)������1�AYc����	���%� �<|��x�!����d@Q6���sp�P��i+ �<8 �\�(�L�d�Ӎ���<���t>n�L��vo�i��Xni�Ï{y+��`��\��Q� ��YT'o����@�TA��{y��up:ݱT���� ��s7�dL�ŀ��VA���S�l- ou��j唰���(�� ����� �k� �Is�m�y)�ou�.619�u�m��2tj��)mB��	���i>�@�B�}�����iZօS�F �����j@�4Yɯ��5.��V:�B��"�#���\V�
k0R�f}
��C`>YL����`4�N��|L���x^��(Ü͵��hp�9�sY�X4��g���9��:O���\_�[�99k��0��ֵI̐;�3*ѫ&�@צ��]���V7m*��(�].z�kC�l���j�h,	��-ksźȋ�:9���@��#嫒3�,��҆���P?��;�V��Ƥ�ZT[�EB/�Ҫ�"���P�.z��ŧ�Y���-0J� �v��GY&bҁ|}�d����hC�M�i�f[O�Je�a3L��!j��l�ղ��W��Ml��i� �Ck2���̜F�Xi�7)�-����=ڸ��o�:}LA5WROu��sdwDQ�A[N��E��e�ɯ��~��<=~y~������O�O//�>��k1�׈�I�6v�vB��H�lD�F���n��.Fv�u��#��jց|�N��L9J��4�S����ԓ����@�j���ɡT�-pn��>xyݣ��("=����b<u�¶J�r�!T�hcbF��[�Mz�:�[n��j6j���)z1�o���--�?����#_Y���z[4������f��M+=%��B���]P�Sc��㞿u��9O`��Xj�4d5}�3Jnf����w�-&�"W��f���A����KD�q�����,<&rN��4T��Ah�l������-�\�3�0�d 8�&j_4^�f�k�z=7�R�M�TB�4v��"=�p#­ޤ�� ���a�j�{�[�jSN.��v�Ӷ�U+f�a.Z�u\�ć��v�h�[d�ncF��LA+���,TUV]�s�F��� V��WD�o��O�5Y�K��u;��?�]�Z��Y<����r�1с�2\u�d�1\5PՁ�(Lڂ�U��-�&7i�еZ����\���4�}L�����e���5�ז?�Fܓ,�u�КQ�r���fu�D�\�V�S#�f8�f%��7�[f�Wf��& �'=�Њ�N�T1I_-r�nT������y��`iT�cZ1HqX
mc�[�V�qܝGhcW��н�b	���k�����f�F�͵��K9Uy�j?��e@(^-�Z�������h�z��v�޸�c�2AX��ҘQ���sn��������	���iu�l���K�g�E�6n�IסD�q�`��g柴���KF/f�Y�&�?]���P@�ބ�Ì��<�=��������,������Q����P����M¬D�ki*�04[)�tx��pH'gJ��0Xd�Ѣ�Fc�]-�� ���o�@�ן_�_�B0÷�J|c=vHkU4LSg����&n�W�Ϧ�D�֌>kU�������NQ0�Z����Ap�g���R�Q��W�2[:��bk<�"��NW	&�:�#�?�m �`I�ԏ���y"�M���E���� 7��׮5L�2��]'wx�[}k�&�0ӽ��I%;ǂZ~[36m�X�:$��p�cA̶5T�������9�ʩKf�G G x�!!��j�t�i*�$Z��<듫���l�[1�^N�_;�t76�;�)t�E�{��M�Wck�^)Y����Q)���߅�n��E|�?����1ːP3;����ehr�Zah���O�e\t��6d=
^ ��B��^h��,�o��Բ����P��C7Y�Qc��d�Zt�/,`���Y@���+��5閰G=^A"��S�h�*#gE�|e�x'��t}��t�p{�#%B��k�E��QS�Z�?6����K�1qYv�����l�h���.ǘ�:z%5�����N/iZ�1�k\�^�����땛�xl6r��2�k�ؑ�]Z�ʚ
�h���ژWMN�_G�v����!�#��mt�F=�^E)� ��aϵ�n^]��m����	�Gz$p�3p�4�9}�,���M"��ȩ"0o���z̝�"S�V���CHZ�آ���X�p
��@R��Mg��՘�m�c�,��=�s[�H�7k�Ȥ��jNr���\�)уU\fNo�x�[�!�4�����7���M��;�/'G�:�Φ����N��K}G&�&���hK|�,��S���O݈��V�D0u��2,V�Z#{k
%`�u�F�Lt�IFûʄ]�~��9l �8��R�9�k�Y�1��4l��d�c�tk��v,�%ff+�n	 Åo��i�V���K�J[��x;]�N�}ٖy�����-1׾��c��3�hLM�יG<�Z�C���%s0�6�fl1d6T)4�;?m+�"�`:���Ө�i���BYy���ʋ^t����m����iKBh��A�]�@�.����e�fzB\�
�ltpq��q��><�O1��o�_];�ċ�1���5�� ���`���9/m�&J v�ƏT�\Yv86�E�4U �G<�Q�<b��)Z`�`�C�&9�m�!��5㕅�+�=��/��馚B��z���e����C�!#�8�b���a��p!�%��=��c*�b �җ�Wp�e��n�
��+���R��Us��mpN��%C����l;�    >���ޙW6	B����%y�x[W���1, ��S~]�%e}��	6E[�l��-�7|e�Ph3g�{��˄iʩn�X�?۪�* �k�&#��Ҏ@+PU�����5�ng���'
;]��}���U����
���.�@�%����g;��|���s�ƥ)�������}i�p��X]�a�M���U�"�p��+^GD9�j�ü�%���8�.���u�@ji��Ui�.ruw���\��V,���}�s�*Q�������cbZ����އ���|�I:c��ƍ�yL�N�6s'�$���#쉰8�+�-�z��i�I(��fR�f�u�
'�=(cF���l8�"�J�.��+XWɪ�ʶq�-�4~�:�w��n�ᝅ������̠�\�)Z�	�v�/0\#B9�=��QuU���#^��sW��%����|�Ybt�n����	o�/�z�r0���/CHmfC�rp�u�;6��6ŉ��ߢ�-�_��P'�z�o���vJ@�9���0���j<���J߮�AM��f�8'�T��ݚ+��LG�K���G�#L5�R5��?���PGD4),���f�P�#"2`"�͂y������fL5�-0p�3k�-a�}��[߈QM��=�����#7%x������fg+�^��o�|�Joj/���՟[&6���*���ݏ��Ly�d©�`*i��l�D���j�U oX��I�Io�$������⟅���h1�?�A������B�u^FN�:/w�$��D&�|U���G�6���i-�[?`w4NRǆ_�ɩ����v�8'��>8�la�����[���������S�,q�`�*;5�
n���w�<��(�b{2V��e��d�/X;r$���p	g�U~A�tW^���z	��'�g��^/��ת�;���Q6`��s-tS�)|2o�kCm�[婻3!�x�O<���d�����83a�	�gt>L�=���:}�Ј����B���=9O��o��������W67��f�{�B�'}YF5´!����Qy`�D7x��Kn&���P�sTH[��dOb�M�57�a���z�C�n�Z=��������Y �a��ʵ����\���C�h!=>W�۬	oD�o9�?�+6��7��2��Prއw���3��Q�+��M�8/��x�C01�<@o�����˪I� �U�3�����n��WMn���o@?8�DM�#�a'��x���U"j�B�� 7�ֿG��^�r�nu���y��p��-�(���Cp1
]�N﮾���+�c���<�"��DQ{(�`0JM0a���Ä�E�#4&��f0Q�<�͡�����װ0s�M��(7Ln�'�=��F	�D���B�ClD=��x~�&���!�����<��j�XY ��ˮ�̯�I'��{0��J�"3l�?L���s[>m���+���s�+Q��TW������b^ʝ��
ܩ�/~�)o�nݲ�(���痗����~�������OO_�?=|��y�a��p��1�]y�IdI\��fzz���{}��k�w�Mv(s����,�곽��R��b���+�x5�sQ7A��Jm�e�a���M*��tjѲwj!_ɉ���2^H���ʘ'^\.����'DD���AO��~������ÿ�����ݧ�S�x΅`J�h"ҷ�A����4����������a��صF���7Ke�]��<�Vg��Ui�ǿ�������!��<�4r-��c���}<����#y�M�����L�>�zjC�Bvg#Z¬=y3�,��D���:��ی��r������\��@Y�X'<���|������>�~}~���#8_>���Y/�:�}
g#y=^@�Y�}u����4��4�f�ԯ�vBy��I��fa����%��LX��4.=����_�������r1���`�ݤ�WH��ChM���RdE:��&����W,ev��Zm?M~�����(�Rb���I@s��� �:7G��z�t#P������J��h���f^�j3;�e����+��=W���se�	i˕k�������яHYw$Rօ�B'o��}��VmBO 1�Z��4��	������/x��C����V�ZbZR�ORڏ�s)���WŽ��c�e��g�}���c��#�H}@�������}�(�}@��!1��~���I���OH�&���?<�|���ק�#��%�
�X����u�d�t�h|��,i�P;o�F�V���,����fp��Q;Bڴ�{��ڢa�)���P�OOx�L��ӓ�YT7��(�o�E7%�A5�=��������������^??z|zyy���_�k�W���I� /�r}*�A~0�'�t��_)s-��k����T�Q\�}���������_~�����������������?�7����������3���o>~����$F(���HxPL��S��)FE:�Wt�)G���)Q&T�l��a���,��1��hv36�{��}�f�6�����]n����l��Ѱ���7��v;>vn��R�%Ɵ�s����e����n/��	9��@D���-��[�!�>�kW'_J��8���qݕ��s�ht1=lo�
�U�iQϑ�H�c���vŢ��fe �\o-��zh���a�YZ���v�J؅��Ul"G��G��q#�R���P/F�������;Ҵ���Ɂ�Nk�Ev��լ�{��h�\��{��n��b��.���Z�3�&�
�'=X}���QS)JpQ��h�p�	t�C��s3l[���?�����o�xJ��UQ��A'G��q!^��6��f�G�muRm�{n`=|<��t�Y�Z�.��]��x���	����`t��#��F*J���BIn|c��`4��;s������1��n:	�;���g9�m�m��+x %����Bd�I�y �rfJ����ס�i)���ɡ'�O�=Q�������P-L�QrFJ(�oE]Pg�P��g���DRb�rё^n(G�1�Zot�~E~��(���8ۚ��Hnc�IwrD��LzD�+�l?�`&����C������ņ�()*5�[dYş��3�d��+�D쉾!}6! �w�<�$�R�ܲ���*��uS@k�q�a]UyRf#ĭT��]_)���!�����p�9B�����!�M+Fؙ�剄DK�2O$+m�U�-N��g�~��B@-�Y	�P���oţu ��r�(��\߻.�7{R���h�a�b!��ֲC�I�,d#��x>+��,�,�b�tD��a�E��/�����✐�<We�����%d�
Û:,�9i�F6DlG>����CH�%Zu���u��}F�:�`��Y	��KK�á5!	^T!�g����r��X�u[;�K�S��V������}>ه��}E��Al]vdxK�=�m�b���i��<"O�ilfy@n�Hz���um�˦eO!�%d]	q�B�6ݜdTwY��*"����_�1�k=��`]+�O�+B�V���^$��L�JR��Y��(	ֱ��Ε�(Y�%��p)i�(��&u8RqG��Z�HA�0�D�l�l�m{5)�jwfJ���7�$�w������P�(���l(ڬ̚�󩂒!_w�GɝH�]Qr7�$��:J��2�',�S�BJr�X()6��N�b��L�ڢ��U��%��ȷ�lg�l�y8���KM�,U�K#�b=��JP5j{e� ˯��FI9J��$7�0�nuD����d���*(k�N�����%���{��N��p2�i)�w�w}��FJv�y%M%�;C*�$��j]Ss�9�4?Gz7����2E��grO�܋�rO��0��
J���͟G����6��J��m����;�X��ڡ��'�C�w�d��E�:\�2:�4��t,�W�I쳹ft��RQ��<�Z(���f[�y�k��K�G����ם����u�Q���$������{rh\��7�DIm9�{�$��Jk.[\�DQ���Q	�DG��U~%`    U��������bf��J:ԚD̓�J��诂�N��n��]��KJ�;��WG�
�PbQ���B���]H ���4;�{��d�������ٷ�Q����C)��ʑ�(�%�(�>|��sB�3�6J�U���`s:�-���&[��pH֌��j0���Bh�yF��%$�����/ϯ#�������d6�&VhL��h|���ٻ5��0�+	I��5yHHEHF�N��yd�o��y�6�Uϥ6=,�^4l�����eO/����"������K�!ei��m�7�r�/B�t�#d����{�5�������e2f���tY�i��+c�?�>^�����c儿^���/��y��ѯ��~��<=~y~������O�O//�>���Cf*�2��sLS�� Sߪ���(^ �=
�����oB��e}�>F��wCaWm��V~kBv+?���@�����ؗ���Ml~`��c-�g{�蚌af������$w�ivYK]	!wa�P�@���益��
u�_�T���_�^!;��� �wO��G����nN����Bm��U��D��'eUGȝ��rw����8�;�+,�A��Y*{�^\��Y���)��`X�Բ�`^_l���h���>Ԇmϵ�:�`�2B��ވ����p�^!ٗ��r/��}���R����!��V�#$�<�� �$m�)U!��D�fmQR!��uu�3��j%�>D�^�������n����`T���)�m���hu����O`����D����=���4[�҅8���2n�ߋ���%		d��wd�G�~O��(�B|�a�l[�����o{�7���EHƨ�j	����^!� ����&���k��J[���H֛fq�@]��8����+��M��]`�ER�8�`s2v[%z[���`5�2
Oh4������U�?H���]�Q�mKe��h3�,�32�&2j}݅V��_U���BHs'��$���:B�]݋h�O�������zAUQ��ߴ%	�mg�m	�vX���������B��� �u�c�3�_j���l$��Q��8���Ĳ�ԃ�eT ����-��g�����;Q:ӫk%$���>B��0�!��mU���F�ʞ��&��S�2Ȩ�y���B��ޒ�8���Cb���E�LhzF�+l��d�̥�2J��"dԙg�$�R=���^ �! T5�d�rk��R���c+�ٜ�l�|�d`�v!��c�����p,�^<�\�g!�����ߎ^�m׆Z��	�^���dc��ZA���`�tIf�L�{)�#��#�B�����R�qB�&2�<�}vtW!����"dx�2��*K���tNn�[�O�=�>���p%�o��w�c�^�P�XzQ*��َ��Pߘr�s�][�f��΁����&���0a8�a_�+�k��K�P��YZ�(��CI AHf�Y!$ӆUGHn$t~B�<�+�e,�R�0a�1aB�����ˆq���1��j0�%�\3��%81�/̨�wCrM(��\��>B�'�t��m���"6���l�R��<6��:��5X!�3k#$��������!�#��#	i��lXVRխ� �.��hFMw��rȨ�y�Ž&'$WgU�|�f��y~yy}x�������w������
�%�'3����#ϱ�H���ٕr��~DI3J��E.��+��D����1'�P"��DD��J�dk�AvV�ސ>d�2���[I*'���3���iM�HG���LUl-�#c�yÁ�þbp��HGA�ڿ҂�e�S%*�$W���Έc���A�<U���3@)�i�i��c��-x������ TeH���0�B�I?�,�6��h���p�A�O�'�N�&Gk9��bd�3l�[, "��
�����M甌��k-�ؐ�k�x�YlQ���d��n�mn�G�t���k���mMdc	���`*�)���Y��&�1�fͨ�M���dI�^��MM䰞X�j��ݕzk��j�=�Y�T�h�^/��t���Y^���v�#c(�4[>��43�����������V�~c,���d-p�=��֧��m5�A� U�k^�!�i��=����+�љ���%��&�T�Q��H�`�`�<.S�WE��M¥�oicڵNsE����M|t�G7�@9P���)j@��u��5pJ�n�Q�9J��~��9d�m���֐e ��������K2�E�&��#��PJ��SS���EJnVc��`�	����(8t]_���J��]�S��G`\�����81[Ѥ��A��'�<w�/#�="�rJ�>,?�����\iM���{��'#�H������1�4W�T�����][fTJ^��a��CM��\�Ȯ��a��rU��ƈ5��}M@0ȯ�һ�������������g���(�?�<�� ��x0��W�vK.��|[�˕##�1+���MF ���/a2X�ʊ�� ���:��	��%�֜�ZGQ<��N��Ŏ�+	zؕ�!�dI���nIT�d����#��[�:��O,u;�0��
�P��-�3�>ڻ�pf���7O�Y-�A��&�q�Q#e%������B(�����Hȕș�Y�.�'c6�4᱋�:	�H���tP\c��.�QՄ�zk���9k�,!��W:~�ȈkfI����n��knN7NB�O�b�.�4ĩx���}�Ț9�"��5,:����8}\��Y{i���`_LyHHXO�m��������q��@��!h��)�a덒�4�.��Q������0��W�39�r#���^Q;Tw�A�v��yD�t4e������<���;���W�@ɢr,m���N�͗��E(<ƂӁ����B�G�ӑ]��<���"m�e����M�1/�վU��~�b�:���swt���%W;�\U!W{���y�@ǎ��v�c�}�����0��|n4H�峚e���Y�ry�<f&�=��6LK�ń��ǡc�Yguб}�Y��5b��}Ё�n$�t�ϦkS��n�J�oׄ1k3+�_���!���3�����<��v����a{�Q47���lO*2��&��q��^�{�aU W�=�>���`-P�SIZ�J-[��1���-c�qAg�ƻ�@w����v�*�#w��7:��ᦣ˾����h2�U�5eU��v.}���(�뎡c�����<H�GM��=�/�����+r����DI:hf�\��Zҡ��;��^��o}���otT@���Rt{9$ӿ����h2�UI:*��{M�/o?"&��;L����OG{ �.�f�Z����NǍr钎�s�-rU��`wcC���ڧ�����}�(u:b�ȍ���xK'��%ѽ��F�r�)�>)�8�2����0޵�dv�T��=�M,�w��X�[�����+K�7ٽ�J��냽�ܼ�(����@8Xay��20X!c���F0�W.�V�(�Ƚ��Q,�$cCS�ڇ��k�9��3��.��f���rb��f���[�K� �����[_��@�j"#��������(�.e ��n�r� c��,>�]$�@��`���Yր�]px�hi b'�.j�����<��ר�Mw�ٗ��=0�OcCE�72
�?���V���x�s��*c�]�V���Ľ`,b�18��S-ų��l�-Y���'��1�:T�dO��䧉F���bɱ�͂l�f�ۇG +�h��-�h�!�0Z�&��/ொ�]V:�@ƎB��h��f�D��*�rd��(��%˺Òe��Ī�v,�/�o�Zխ5w%�{���;5�"p-2.�tA���4��#h[���:ٛ_�������m�e�狼�i�\>��Nz�uO$ت|�[?.�NK�\����d��u"��qe8�@̽*�4�k��^u��\��r�3ӽ;������{�ڸ�)�2J�|�����qGw����F0w��/�j�$B2D
��I�F�@�gw0iGl\��(8 ;�.ĆE^5�f��EƎ���#    cǕ?�$#�RUG<�~���}ɸ�� ް׼��[��vv��	�:��ο��	ʱ:8� �/$�����S`��9�7j5���~����%VwPJ��6�{1d܃N��Ak��9�	��$#����#nmiU:ǥ��:��
A��]����0��>^�5��*2�����q���̙l���\��c ����A.��(�22r�����_gR��ݰ����2H�/GFU.�����od�9�5�q��d���&@FS,-#=n=7��F� c��	�������	NF~�c�$�`2��}ߞF����EV�m �ߵ�m���D�RM�����Q\���^S ��K�kz���tI�����ۅ|2���rd7����V�PH~iLU� C����/�f�Ɖي&Uq�.`#ǝ�G�8�T&[���,@�����କ����J�:�Ef�kAs�I'��ɋ��$�0�{�m�]$�h�eVh��W
^���,d�LbȀ��5y&4�fe�RB)[9��eэc���8b�Y�P�pP������=F�A,�h���M
n	)��+��9@f7�F��x�HGt.�mJ��9��m2�j*��dX'��m��N��D�{
�j�\���g،�i��9�*.�~:��, �A�
'���l�aG�=Tr���`z?���Q�-#�_c�X�n����d�N�J"��zOAFT��2��D�&_�q]ݗp�9�#�^�Yd��p�;��}f7,�檍	2�g���==_�>��t���?<=�z����bo,��iÐ��"d�S�12]���U�MC�xg���uE&���u�0���r� ?�<�R�x)1���s�����v|�V�Q�����O������I@#�8�R>!Y���백]�&F��`$k���8sYB��O[+�"��[7d��loj/.-����fK)ֿ>��槇�߿���>o���q��;��y�
�c��I�#r�������&�E��U�����:����+Ӳ���|�eU��H6��ϴT�9���,�/NG0�o�'ޮ��r'84����)��0*p�9T�a�$b�R�X.��b�뒨i�C����?=^�?=|��y�;��9r1�
"��:�G4`^�5m�p|�YBM	1����y�7ڜP5J��1������߰k	�Q�iĐ�D�V�Rԇ��C{��n��a�	G7'�!ٍR�@�]���P��A�I.u��rAJ�\&WZ�ɕ`W�>ΟY��>Y^0h/J���4��,�E�E!��"��.��Léݲ��D�:�_q�C�Y��e{,-m����:��@��>��iZr��M��G�\&��#Ɏl��4b�"b�9O"�f���%���3�a���������M��V�����޹nn����8���}J��uN&S�3P6>��� .f���=UQ5Q�{j�0��p�[P�zx�a�BE�3m��<2����V�G���	��rZ��KI(Y
����a�S�lkò�����x���Ⱥd�q3<�e�r�+��3W*�3��}H��F�У�$=���<��'�.w���T��`e�
��2��9�;�y�k�������,ů c��  큒�%#%��N���M���a� ��
B|�W�Nqہ�(���ڈG9F��`�4��y�lL����[9J�ֿv<y1��69vU�BaqC���GXXh��R2n0��"*H7^��̵��1w�C�.I�i�m��1q��Ae��1l���P�.�{z�������߿>|�������y;\i��j�c���������[�̑��f����```*8���t̮�� �n0#4D�l�՞��P��ApF���圴�O*1�ç�8�6�M-Y�x�"���ZI�wz��Ȭ`��᪳M&םc��M�*����䐻T1��,�p����[���X��Qq!�������ۯ"D��FݚM2��	�=��q9��������u�g���T(�QGnI-��ohiI��4f��)��&�s���[z�[G���U�٨�Z?0%!'��l�aq1	l7���9*��H��2�S�R=*�ڿo��X�L��&��hkܷb��$jq4�m$V}�d�*n����0Yʠthw
�I��������_=���K�e�Y��c?j�u��C�M�Ŵ֯D�Z��R c�՛�߾���a�ƴV�9�Ӌ���9�t&��-px�-J���܂��ES}B�`cw�^O�1l5�AvYZ��f�@�k�Λ��*���K��-���E/1$�ԩ~U�x5�6r�@����0�Ì��zK@�����ɕ/f�?�	/=�����Y�?�H� �J��*���ժ�G�`���-�uq�t�(�>��n��H^��k���(�=.O���h��4��������@ J������J0gO��w@.�c�z�٭/6��2iM8
,e��^�?�%S9񈙣W[C�#�P�LI�K���AXR����d݃����h��kg��5�GNu���Z��H�O�wU�D�j���m�F��$�d�*y9;��-	�:�r��"&�$g�5lǓ� +C���(-}�Q�Yb�y��]���8�����J:G���S5��W�[{�Nv.�`�R��9��R9���W��9��>*^�jH���`�v(�����u�Cbwƌ����e��J�ᴮ�mqR����t/�pyk��vъo�'{�u����U�N��O�괖��&q; ^1��/�ۢ*��]�u�5k���薬N��j/.2��]�B��u䰫�ꁘ�TJ��M�ǭ�=	��:9'H}���'���Y�mU�%S��w�쀘��Tǥv5l�@�ƪs��x��Q�H�Q�pZ�v���ӣg8%j ym7t+����;�?��z�Ih�.����[r�n$�`�� 9�ж�zVϬ�����+�ɩ���j��?���}7	�������[
)q6��V�8���+�{hb8U���PC�!���.��9�yv���D�JJ�?�*��}rU����"0�Ձ�'���wwH�̜���G�E��)F��	�*]���.����1�2�O����8��v��&���x����������g�pJs<fN��k�C��(I^`)k)yi��4�����
��X��|ZT��$���L?^�#��-�8~��$B8�+Z�����{}�)���g�D��|�׆-PB5�P��(��{�'ʠ�C�_⟆Lj������`ι)�*�&���lSq�"���oY��6�6r8>�1a�F�ᇷ����6C1�l!������Ue��&#�l���DO�;N>2��%�A��;�8+y��7�;�߲u�
�I�f�7]\�xv&��i�Q|ԇ��>Й	/e�����W==i�]/xj����[���8�����崍<���䍈uID�X]������Q�>3.�L:8U����Q��X�Q	�`��7d`��	�qf�'�++��jB<OW�Ȥ'��>K���?�Cv~�gN1ǖ,8���4/s�y��	�Oj����F��.$�va���\'�ܩs�����2g� �>;��Gs\��8�<l!zk$'�D�F����:��x�ڸ�?Ǣ
�4������ը6�q��b�|F)T2�$�ů��R5r�tj�f�C-��@!�B�t[���
r���0'��rk�-S�j@��,���Af�Ҟ�ͫ��疗�~�г��Rm�7fgn�~˪���ë��멎�=
��S�ZT��Ի{kf�<��]_��&���|�:SS}8Q�o��%�!q?:>̙qK����s|c^�*���o9	q�l@9^��;�v�.����zS��j��G��v��?7�O�>tsW�>���6��AE��nV$�}?���
�cC,�9,���eP`�����i����f�~ɯ�^��s�`Aɫ��1>s~�VwW��G������џ�>�S�Up�2�[�o��ħ���Ԟ�?���9�Gj����3;�{gn���SVsS*�y
OJ.���Z�*;��tl}rΉ�!����C���;MJ �  �>-ɏ��ii��}Z.Ğ5�7'۴{�*7yUUI��%H���>m*pҞ:��2��8ށ�21��b�$�Jʭ��专1��j���Xv��O���Pz��� ��m�b�i���}j"��z��r8�ޮ3��m���Eȃ��U8�pM�Co�pQiM~������W������OtH*j��7�iC�}Э����6��q<�'�}�B��`�����WL�>犓g��$8�V�n��+?�<5�2��Xu��b޺џB�|V�qvbyA�0b��o�}\���:{6n��R̟x���5���u���7�"�i����%}�"<�⮝ ����5�7���&�Ƀ�jC޾Q*a]�4����W0����� }b�nm��i��v�M�Fl;�^��7� '��V%�s�j����[��ѳ���f�Թ�<�ӣ�ͷGIs�������BK�+�w�)2�͟��)�}��I[������&���� �g�����|��P�O�}I�m����O�%7�T���_���*�u���f��ؿ�R'�lM��O�ҩ�}X�OL��ߔC߆����]r����Z+[e�1N�]U<y�tm���a�$�)!�3��:��G�Tv_��$V�����{��Г����\�G�o� �$��LN=&�$��V�����t&O}�}j��z�|9��}`�s�'�����`��S�����>�p��R�\�\��Z���ۋ!3�<��-�pr��ϱ��ʜqԫ����[`�����T8I/���%e!��-�[�k�8��>�O|����7Ċ��Xq����
.`m(+���*\��vX�:���w�a`Xo�2��F�N�p���x���|<���>(J�T���������>�ֆom��l�!�r9���!�RN��{$X6ZE4�<�2q�+���Ju)y�J�r��s��%����ꓡ_϶�uX�j�m��%��)�&����,Iz-z���ﷸ;��۵� ��at�qDA���-џx�&I�Q�4]^�p�Ύ�9~�����\�¿�A���%�6�6��&�
=��q�(�����k��O�a���i��kC��>~����燏?>]����OϿN���|�f�o���	��c)(�N#:.i��I�ӣ��ˁ��B΄3m�6��VE���k?*C�O�>�������׿ꐂv���(zۤ�p5���y�q��ο���SK&'�_@�{-k�xLצ\nm�Zs'B8��p�6�o�j�
�8�=N*ڰT��S�}��ǡ�1��
*C�X���n��>*��ćMl�Ή���*~��!�8X276��4N�sU �a"��P��؜z���џ�F���f�&��JA'��=��B�8�����'���Lm. Htp��m���t\8�C�_⟆9Ε$?���>�r�g9�x�+�ϝ\����d��7o�Lh��	��Us�t���Dl1�ۧu{�";O>�%���k��т�Z�݂�X�1��k������_��f���o0H"��׽X� �P�M6!������w�xon�ʱG!m����c/�V\F����L���ۃrx��v�g�3��� X�9��t���̯lX��G�3�X��O˜�A3� {g���,�q�Պ@Sep�^q��܀�5�
���X���d�`�0P[�?<�5g��Ė��<G&AQ�_q���H�Vk�Tۨ�`6��ލ�$n�K�	,�u��0���b��^�Þ�+����B��b��m�,~�:
t���}&����#�I�@Fm+5*�W�b��6��;>�4_=Se!�z�"럎F;U�IAiy��lj���N�©�G�SWo��RW�Y8��â w��c8 {��r��M�5Hl�f�Ä�3q`X�$�&����u��"�uH�B�����xm�U�BtS�a{�29�n�<ͦ�-���ﰬ:s�(!�)�J)�U^�^�=����ѡ@�=%���j�ea	�W��o�M�/����_��C�t����_������      �      x������ � �      �      x��}Yo$ɑ�s�W��P;
��(�lvwAuɚ^4�լ�]I�FX̿_?�2�#<�#U�b2Y����]nbo	}�������������?�6��������^�������߿{�,�z���K�/�����:�����������$�����o�������ѥ�/��WL�5���_I?�j�c�����o�_;�0�2��r�/-� �=���v�ז�A�f����p*t��y��wm1�|��/P2[�bBݵE����k��/B��A>g����s�����8��p!ׁHִ������/T|���3ti�&����ȵ��h���?AW���7��$�#ȡm+�`�G�3��'^��LB�{�
�-X�������w��S}y���ۗ?�7�$�%W��g.�����].���W��t݅��S�qF#���\ۖb��V��߀?�V�R���o!�-����J��Iz����u���8���i�N�����0�I��p��IS��	L���L8��D�O�ی?Y���
�v����
)ي4,�[Ic�K��讄cП�I#�S�õ��I>5!��w��IZ;{j�<5~������¿j�zj�|j����ߪ!�(�9X~dbA�k����_�=��N�݉6���O�Ii�H�~l\�4�c�x^�?F�cmx�D1��0��J>�m����(���C4��ז6�$Mh�F��i��)�(?S��a�H�oEZ?��B�A�����~l�i�.�Ќ6���$iÊ6��4m�"��v��	�c�L�O�)��H��VDh�mEҡ-�dI�=�I��i�､�X�c��(��M����|lV�Ȋ��R�	>��%|�������O��}y}x~}z����:n�;���+%�)�"���q�0��-7B�#��Szŭ���+A���A��NpG��O�{Kc�r����,���V�
x�`�����;��K�y��J5��7$º�����	��	��ʱ_����ߔ(+��E�� R��r�"1]<AN/�PfA�|����t��(��|
=��4�҄kЛO��Ҁ<�K6#i�Z��t��q��Թ�*Ikie�`�̀+Ҹ��B?rZ�/�?��ۉ��|�V��h�8!a��'I����ܑр��%���ZC�YI��r�6.�'��F�(+K�H�O͐�	wf����&#��iX}j.àu�0z&h��F��'B�¸��̷xfX�a�;���0���L��0�aDo>�g�=�Ʉ���Wm�H�����M74��T8QoX�I�:�0�	���8Q=3���d�a����	�?vz`Q��B�;�6�,���?��m�(Tbc>����ߋ04�����0-��-]��ׯ_^߽�������������~}����7���2'I��o/ޒ�mk�wB�HR��=7�<�e�������v����Ɔ5�<��D�����?�`��[��b�A��̪WMLGt̮t<1���b�A~b:�s<|~����Q��	w�{�j�����W?]}tW����XD�o*ĄG<�o�.�(g���ɠ/@��;�q�Ҏ����"�D{Q��V�JWI�{�w�3�	5��pڄa��QZ��AO/$�2"qR��D�J�*OS�`N��ߞ��>���ǧ篂������*�å��W��1�,C�0�D����T�.�k�hҕ�h��U4��wq*�����'`RQ��R��[
!2&̜Щτ��U�2HXd��*���k��\ks��{{-��:�h��M?�3���b���%$��V脷"<"U���S�*,mPU�[	��/>>�����#�2j(���_e�I],\hC��`*t�8#"O"�0w�� �܈@	��Q�w���JIk�տ�:^=�����%?<^8Z�_G����N?�^�`I�L�I.���/'"���t��ʔ��<\D�'3���Hڨ~DH� �r���_;Î�X?3��:N4�nh�gɳ�i���ih���k� �<�s��p�����]/^1^Р#o����$Ov6�ĆfnAE5����W�U(&���R���$�U�i)Q"�,��(��B:�7�T�d��e*l��L~�7��ՇM�C:�L[y@�+�l��&B�;��vH�#ڕ�(O�6B)%��[�B��X06fL`�Ώ�R=l��ϑ��:૸0��k'�=�o������L�ͯ�k�̍�`A�x�Nsyt���0Hs���R�9nn�<��|~����ÇQF)��c�RB�Q�Wr1�����v�k7�.GQ(�k�(�5q����i�ʭ�iHm>j=�@0�q%ԃL�_x�q��j�*C"G�
�;LY"�n�o��������]�g>?�k����"רl~�~:���6�ҵ�"�?�*�7�7i$x�D�/�$�k�Q|�W,2������ڴ�@o�Pf�X4N�2#���L���A�eF3	5�?�W �O�^m��&M޾��(zM��٥�kJG�A+�p�iEO	g#�c���E���AR�aM@o�a�yG* �hM���Z�h���VM��A@ű�h�]����Q]�7�l�	0����W�ܺ|F���r �둔 �X�aDJk��B��GW�2E\:�5<͍�yA�z�S]�gk�o[�uF��$o�/"'!x�$�vH;�=?Fᑮ�pM��\_�{�_- @8��sw���^
���^�^n�����?d�^qx�1�Ӷ#���U)/�ΊA�?��Z�(�����$|N�6�L�~A�|7k�ݨz�~�|� |�"��2e&��Z%\$Zy�,��*W%��M�E�p,ŋE��Kw��XjFD�ߘ`����e����"O��#�19���~*���8�!��+f��?�L� C+q�:����K?�k2$��2��>QW��kL�!�$�DŃ�ы+-����.����`���Ӊ�A*zy�E4-Ĩɼ�?��t�(3i���|��4�.��\�U�`�c�I�h
������oE�X��:�!<��L��n)�ǣSAO��A�]�2���%�h�[�t��]T��: �U,��^>�>���o�~|x}����
��U2^�f�+ﴗ�,����</"�VOK�RT
��e�i�g�Y/��w���toT�EcD�3���� W���a���>K/G&O��ۂ�K+p  r���ee�	俑qvgȅQ�x�D�����J6o
�$<�I%��� ��#phd�fЧ��n��'��R�Sp'����_�z��������������ǧ�){�;��"Z���'�� 6DZ�1�6��m�]����痗ׇW�N�.Bau��H�5'�^��k�^��7�P�~	�lQ}�$�߂�p�@�ŵ�wV4�),y%��+ۿP3S�C�j	�'��7�!���;�7t��~	��n���a�&�Ӧ�H�|�Z`��I��#���G���~�i��a@`�K��� ���5#�P|Hju&r��L3T~�2�Ngo�T�J%N*F����I'�i�BQ�k3��L�IZ��q��NگBc�[^��K�ә�Qo{m����/���W,�<���p�%��|�D~(g�U�}A�Q�0o�(PҲ�$�1�4����^H�Hɨ��	��/W*"b�!�,V�Q��JiU�+鴊��Fe����Bp?�����6B�I'�2B�����k*kP��}��c��}+s;⬾i��e���#n�03���\�W�H(
yש0vT�����j�԰Em@����ҧ�H{.��|��,
�g��L1��v`���!�:�+p�˔�Ƣn��a;If˻_������=Q��c</&��-����х0\�Q�Jx�Bg�l�����ԉ�ڜMr2YS�o$��DU�L�vZ�w�����uS�����������a���/�F���N!.���k�7�n̋b�_����zP�'q�eE"ߋ�5��r�GH��qne���"��)��R��Jc(��R�U�p^�3	�JN��S��>�U^$~K�v��҉߻��Zz�E�OnD-�@�l�'q    ��q�����=(m1����O��dKO�d��%N��}�ݦ�W�t���.0Ѕ
�p4��	c&3�X]e\3����C&=e�Fϊ��k�A�v�����?������������|��3���_���u<��S�Ⓘ�b����$c��p��wh��y��`���?�7��X��8f����lv�{���a�cS��A:(.��]�!�J���]G
��J���a�$��H��iW�����l�p�tX��U��yډP7��%K��/��>~���D
W�R[�f�/����BV.�"&?��[%P��R�EE��|iދ(�t]Gb���Fƣ�V�M��6�G*�9ʴ�Xj���A�	{��
V�L���X��{�)�q�lW�cER#nB�d�M������L����8\H2�����c�u��:�C3p0��`�����w��t�m�O�=��þ����v�
��\f#NyYY�l���α}�Y��Fzة�}�����.Sr�dg�c�G��5ʢ�zP��ɮ�
yRv��C���mծd����	� }��:���&��qS�g��$��B@|B����N�L����+�6!�WXy�n+\��������//�����o����2?ի���73ɱ'_��Ѻ48=��u0�&��R�?K�~�#�R_�������!���`U�r�Ws�	����Vx3�ʟ`U��-Ҡ"�5~Y�!:�.Dd�|�;q�K�0�׿�4��nle����ק�3����ȊԔ�c�n�N	ΘOC�ce.}䍚y�ݬ���7TSL�e�>5%�`](G��.l�r�ا���RGj���Ǯ����#�Ti�D��kwû�LJqW7������[�^���"�%�w�����	�,��D��4i��F����=�&���o��2��7D���)*od�_ĂW�#0f��t��q�aW���|Y���S��Ώ^��YD6
_fG�`���YL>� �ɉ���`ĪW�8im�fR*�42�ڏڞ�ڞ����ʋ�gb���@r��MzT�����6ߗ�����E��7v�h�PLg�b6��Ө;1w�=�B&PG�ģE��s/i�u�k�� o�����z �Nx�%A���>���֜U�urҩ�m ��!�.��f �q2hDDC�Lq5�	NwI�!�=�����Y��Q�t��
���k�	d1�'��pғ��G��&�,� �[��n�sc���q)������1�AYc����	���%� �<|��x�!����d@Q6���sp�P��i+ �<8 �\�(�L�d�Ӎ���<���t>n�L��vo�i��Xni�Ï{y+��`��\��Q� ��YT'o����@�TA��{y��up:ݱT���� ��s7�dL�ŀ��VA���S�l- ou��j唰���(�� ����� �k� �Is�m�y)�ou�.619�u�m��2tj��)mB��	���i>�@�B�}�����iZօS�F �����j@�4Yɯ��5.��V:�B��"�#���\V�
k0R�f}
��C`>YL����`4�N��|L���x^��(Ü͵��hp�9�sY�X4��g���9��:O���\_�[�99k��0��ֵI̐;�3*ѫ&�@צ��]���V7m*��(�].z�kC�l���j�h,	��-ksźȋ�:9���@��#嫒3�,��҆���P?��;�V��Ƥ�ZT[�EB/�Ҫ�"���P�.z��ŧ�Y���-0J� �v��GY&bҁ|}�d����hC�M�i�f[O�Je�a3L��!j��l�ղ��W��Ml��i� �Ck2���̜F�Xi�7)�-����=ڸ��o�:}LA5WROu��sdwDQ�A[N��E��e�ɯ��~��<=~y~������O�O//�>��k1�׈�I�6v�vB��H�lD�F���n��.Fv�u��#��jց|�N��L9J��4�S����ԓ����@�j���ɡT�-pn��>xyݣ��("=����b<u�¶J�r�!T�hcbF��[�Mz�:�[n��j6j���)z1�o���--�?����#_Y���z[4������f��M+=%��B���]P�Sc��㞿u��9O`��Xj�4d5}�3Jnf����w�-&�"W��f���A����KD�q�����,<&rN��4T��Ah�l������-�\�3�0�d 8�&j_4^�f�k�z=7�R�M�TB�4v��"=�p#­ޤ�� ���a�j�{�[�jSN.��v�Ӷ�U+f�a.Z�u\�ć��v�h�[d�ncF��LA+���,TUV]�s�F��� V��WD�o��O�5Y�K��u;��?�]�Z��Y<����r�1с�2\u�d�1\5PՁ�(Lڂ�U��-�&7i�еZ����\���4�}L�����e���5�ז?�Fܓ,�u�КQ�r���fu�D�\�V�S#�f8�f%��7�[f�Wf��& �'=�Њ�N�T1I_-r�nT������y��`iT�cZ1HqX
mc�[�V�qܝGhcW��н�b	���k�����f�F�͵��K9Uy�j?��e@(^-�Z�������h�z��v�޸�c�2AX��ҘQ���sn��������	���iu�l���K�g�E�6n�IסD�q�`��g柴���KF/f�Y�&�?]���P@�ބ�Ì��<�=��������,������Q����P����M¬D�ki*�04[)�tx��pH'gJ��0Xd�Ѣ�Fc�]-�� ���o�@�ן_�_�B0÷�J|c=vHkU4LSg����&n�W�Ϧ�D�֌>kU�������NQ0�Z����Ap�g���R�Q��W�2[:��bk<�"��NW	&�:�#�?�m �`I�ԏ���y"�M���E���� 7��׮5L�2��]'wx�[}k�&�0ӽ��I%;ǂZ~[36m�X�:$��p�cA̶5T�������9�ʩKf�G G x�!!��j�t�i*�$Z��<듫���l�[1�^N�_;�t76�;�)t�E�{��M�Wck�^)Y����Q)���߅�n��E|�?����1ːP3;����ehr�Zah���O�e\t��6d=
^ ��B��^h��,�o��Բ����P��C7Y�Qc��d�Zt�/,`���Y@���+��5閰G=^A"��S�h�*#gE�|e�x'��t}��t�p{�#%B��k�E��QS�Z�?6����K�1qYv�����l�h���.ǘ�:z%5�����N/iZ�1�k\�^�����땛�xl6r��2�k�ؑ�]Z�ʚ
�h���ژWMN�_G�v����!�#��mt�F=�^E)� ��aϵ�n^]��m����	�Gz$p�3p�4�9}�,���M"��ȩ"0o���z̝�"S�V���CHZ�آ���X�p
��@R��Mg��՘�m�c�,��=�s[�H�7k�Ȥ��jNr���\�)уU\fNo�x�[�!�4�����7���M��;�/'G�:�Φ����N��K}G&�&���hK|�,��S���O݈��V�D0u��2,V�Z#{k
%`�u�F�Lt�IFûʄ]�~��9l �8��R�9�k�Y�1��4l��d�c�tk��v,�%ff+�n	 Åo��i�V���K�J[��x;]�N�}ٖy�����-1׾��c��3�hLM�יG<�Z�C���%s0�6�fl1d6T)4�;?m+�"�`:���Ө�i���BYy���ʋ^t����m����iKBh��A�]�@�.����e�fzB\�
�ltpq��q��><�O1��o�_];�ċ�1���5�� ���`���9/m�&J v�ƏT�\Yv86�E�4U �G<�Q�<b��)Z`�`�C�&9�m�!��5㕅�+�=��/��馚B��z���e����C�!#�8�b���a��p!�%��=��c*�b �җ�Wp�e��n�
��+���R��Us��mpN��%C����l;�    >���ޙW6	B����%y�x[W���1, ��S~]�%e}��	6E[�l��-�7|e�Ph3g�{��˄iʩn�X�?۪�* �k�&#��Ҏ@+PU�����5�ng���'
;]��}���U����
���.�@�%����g;��|���s�ƥ)�������}i�p��X]�a�M���U�"�p��+^GD9�j�ü�%���8�.���u�@ji��Ui�.ruw���\��V,���}�s�*Q�������cbZ����އ���|�I:c��ƍ�yL�N�6s'�$���#쉰8�+�-�z��i�I(��fR�f�u�
'�=(cF���l8�"�J�.��+XWɪ�ʶq�-�4~�:�w��n�ᝅ������̠�\�)Z�	�v�/0\#B9�=��QuU���#^��sW��%����|�Ybt�n����	o�/�z�r0���/CHmfC�rp�u�;6��6ŉ��ߢ�-�_��P'�z�o���vJ@�9���0���j<���J߮�AM��f�8'�T��ݚ+��LG�K���G�#L5�R5��?���PGD4),���f�P�#"2`"�͂y������fL5�-0p�3k�-a�}��[߈QM��=�����#7%x������fg+�^��o�|�Joj/���՟[&6���*���ݏ��Ly�d©�`*i��l�D���j�U oX��I�Io�$������⟅���h1�?�A������B�u^FN�:/w�$��D&�|U���G�6���i-�[?`w4NRǆ_�ɩ����v�8'��>8�la�����[���������S�,q�`�*;5�
n���w�<��(�b{2V��e��d�/X;r$���p	g�U~A�tW^���z	��'�g��^/��ת�;���Q6`��s-tS�)|2o�kCm�[婻3!�x�O<���d�����83a�	�gt>L�=���:}�Ј����B���=9O��o��������W67��f�{�B�'}YF5´!����Qy`�D7x��Kn&���P�sTH[��dOb�M�57�a���z�C�n�Z=��������Y �a��ʵ����\���C�h!=>W�۬	oD�o9�?�+6��7��2��Prއw���3��Q�+��M�8/��x�C01�<@o�����˪I� �U�3�����n��WMn���o@?8�DM�#�a'��x���U"j�B�� 7�ֿG��^�r�nu���y��p��-�(���Cp1
]�N﮾���+�c���<�"��DQ{(�`0JM0a���Ä�E�#4&��f0Q�<�͡�����װ0s�M��(7Ln�'�=��F	�D���B�ClD=��x~�&���!�����<��j�XY ��ˮ�̯�I'��{0��J�"3l�?L���s[>m���+���s�+Q��TW������b^ʝ��
ܩ�/~�)o�nݲ�(���痗����~�������OO_�?=|��y�a��p��1�]y�IdI\��fzz���{}��k�w�Mv(s����,�곽��R��b���+�x5�sQ7A��Jm�e�a���M*��tjѲwj!_ɉ���2^H���ʘ'^\.����'DD���AO��~������ÿ�����ݧ�S�x΅`J�h"ҷ�A����4����������a��صF���7Ke�]��<�Vg��Ui�ǿ�������!��<�4r-��c���}<����#y�M�����L�>�zjC�Bvg#Z¬=y3�,��D���:��ی��r������\��@Y�X'<���|������>�~}~���#8_>���Y/�:�}
g#y=^@�Y�}u����4��4�f�ԯ�vBy��I��fa����%��LX��4.=����_�������r1���`�ݤ�WH��ChM���RdE:��&����W,ev��Zm?M~�����(�Rb���I@s��� �:7G��z�t#P������J��h���f^�j3;�e����+��=W���se�	i˕k�������яHYw$Rօ�B'o��}��VmBO 1�Z��4��	������/x��C����V�ZbZR�ORڏ�s)���WŽ��c�e��g�}���c��#�H}@�������}�(�}@��!1��~���I���OH�&���?<�|���ק�#��%�
�X����u�d�t�h|��,i�P;o�F�V���,����fp��Q;Bڴ�{��ڢa�)���P�OOx�L��ӓ�YT7��(�o�E7%�A5�=��������������^??z|zyy���_�k�W���I� /�r}*�A~0�'�t��_)s-��k����T�Q\�}���������_~�����������������?�7����������3���o>~����$F(���HxPL��S��)FE:�Wt�)G���)Q&T�l��a���,��1��hv36�{��}�f�6�����]n����l��Ѱ���7��v;>vn��R�%Ɵ�s����e����n/��	9��@D���-��[�!�>�kW'_J��8���qݕ��s�ht1=lo�
�U�iQϑ�H�c���vŢ��fe �\o-��zh���a�YZ���v�J؅��Ul"G��G��q#�R���P/F�������;Ҵ���Ɂ�Nk�Ev��լ�{��h�\��{��n��b��.���Z�3�&�
�'=X}���QS)JpQ��h�p�	t�C��s3l[���?�����o�xJ��UQ��A'G��q!^��6��f�G�muRm�{n`=|<��t�Y�Z�.��]��x���	����`t��#��F*J���BIn|c��`4��;s������1��n:	�;���g9�m�m��+x %����Bd�I�y �rfJ����ס�i)���ɡ'�O�=Q�������P-L�QrFJ(�oE]Pg�P��g���DRb�rё^n(G�1�Zot�~E~��(���8ۚ��Hnc�IwrD��LzD�+�l?�`&����C������ņ�()*5�[dYş��3�d��+�D쉾!}6! �w�<�$�R�ܲ���*��uS@k�q�a]UyRf#ĭT��]_)���!�����p�9B�����!�M+Fؙ�剄DK�2O$+m�U�-N��g�~��B@-�Y	�P���oţu ��r�(��\߻.�7{R���h�a�b!��ֲC�I�,d#��x>+��,�,�b�tD��a�E��/�����✐�<We�����%d�
Û:,�9i�F6DlG>����CH�%Zu���u��}F�:�`��Y	��KK�á5!	^T!�g����r��X�u[;�K�S��V������}>ه��}E��Al]vdxK�=�m�b���i��<"O�ilfy@n�Hz���um�˦eO!�%d]	q�B�6ݜdTwY��*"����_�1�k=��`]+�O�+B�V���^$��L�JR��Y��(	ֱ��Ε�(Y�%��p)i�(��&u8RqG��Z�HA�0�D�l�l�m{5)�jwfJ���7�$�w������P�(���l(ڬ̚�󩂒!_w�GɝH�]Qr7�$��:J��2�',�S�BJr�X()6��N�b��L�ڢ��U��%��ȷ�lg�l�y8���KM�,U�K#�b=��JP5j{e� ˯��FI9J��$7�0�nuD����d���*(k�N�����%���{��N��p2�i)�w�w}��FJv�y%M%�;C*�$��j]Ss�9�4?Gz7����2E��grO�܋�rO��0��
J���͟G����6��J��m����;�X��ڡ��'�C�w�d��E�:\�2:�4��t,�W�I쳹ft��RQ��<�Z(���f[�y�k��K�G����ם����u�Q���$������{rh\��7�DIm9�{�$��Jk.[\�DQ���Q	�DG��U~%`    U��������bf��J:ԚD̓�J��诂�N��n��]��KJ�;��WG�
�PbQ���B���]H ���4;�{��d�������ٷ�Q����C)��ʑ�(�%�(�>|��sB�3�6J�U���`s:�-���&[��pH֌��j0���Bh�yF��%$�����/ϯ#�������d6�&VhL��h|���ٻ5��0�+	I��5yHHEHF�N��yd�o��y�6�Uϥ6=,�^4l�����eO/����"������K�!ei��m�7�r�/B�t�#d����{�5�������e2f���tY�i��+c�?�>^�����c儿^���/��y��ѯ��~��<=~y~������O�O//�>���Cf*�2��sLS�� Sߪ���(^ �=
�����oB��e}�>F��wCaWm��V~kBv+?���@�����ؗ���Ml~`��c-�g{�蚌af������$w�ivYK]	!wa�P�@���益��
u�_�T���_�^!;��� �wO��G����nN����Bm��U��D��'eUGȝ��rw����8�;�+,�A��Y*{�^\��Y���)��`X�Բ�`^_l���h���>Ԇmϵ�:�`�2B��ވ����p�^!ٗ��r/��}���R����!��V�#$�<�� �$m�)U!��D�fmQR!��uu�3��j%�>D�^�������n����`T���)�m���hu����O`����D����=���4[�҅8���2n�ߋ���%		d��wd�G�~O��(�B|�a�l[�����o{�7���EHƨ�j	����^!� ����&���k��J[���H֛fq�@]��8����+��M��]`�ER�8�`s2v[%z[���`5�2
Oh4������U�?H���]�Q�mKe��h3�,�32�&2j}݅V��_U���BHs'��$���:B�]݋h�O�������zAUQ��ߴ%	�mg�m	�vX���������B��� �u�c�3�_j���l$��Q��8���Ĳ�ԃ�eT ����-��g�����;Q:ӫk%$���>B��0�!��mU���F�ʞ��&��S�2Ȩ�y���B��ޒ�8���Cb���E�LhzF�+l��d�̥�2J��"dԙg�$�R=���^ �! T5�d�rk��R���c+�ٜ�l�|�d`�v!��c�����p,�^<�\�g!�����ߎ^�m׆Z��	�^���dc��ZA���`�tIf�L�{)�#��#�B�����R�qB�&2�<�}vtW!����"dx�2��*K���tNn�[�O�=�>���p%�o��w�c�^�P�XzQ*��َ��Pߘr�s�][�f��΁����&���0a8�a_�+�k��K�P��YZ�(��CI AHf�Y!$ӆUGHn$t~B�<�+�e,�R�0a�1aB�����ˆq���1��j0�%�\3��%81�/̨�wCrM(��\��>B�'�t��m���"6���l�R��<6��:��5X!�3k#$��������!�#��#	i��lXVRխ� �.��hFMw��rȨ�y�Ž&'$WgU�|�f��y~yy}x�������w������
�%�'3����#ϱ�H���ٕr��~DI3J��E.��+��D����1'�P"��DD��J�dk�AvV�ސ>d�2���[I*'���3���iM�HG���LUl-�#c�yÁ�þbp��HGA�ڿ҂�e�S%*�$W���Έc���A�<U���3@)�i�i��c��-x������ TeH���0�B�I?�,�6��h���p�A�O�'�N�&Gk9��bd�3l�[, "��
�����M甌��k-�ؐ�k�x�YlQ���d��n�mn�G�t���k���mMdc	���`*�)���Y��&�1�fͨ�M���dI�^��MM䰞X�j��ݕzk��j�=�Y�T�h�^/��t���Y^���v�#c(�4[>��43�����������V�~c,���d-p�=��֧��m5�A� U�k^�!�i��=����+�љ���%��&�T�Q��H�`�`�<.S�WE��M¥�oicڵNsE����M|t�G7�@9P���)j@��u��5pJ�n�Q�9J��~��9d�m���֐e ��������K2�E�&��#��PJ��SS���EJnVc��`�	����(8t]_���J��]�S��G`\�����81[Ѥ��A��'�<w�/#�="�rJ�>,?�����\iM���{��'#�H������1�4W�T�����][fTJ^��a��CM��\�Ȯ��a��rU��ƈ5��}M@0ȯ�һ�������������g���(�?�<�� ��x0��W�vK.��|[�˕##�1+���MF ���/a2X�ʊ�� ���:��	��%�֜�ZGQ<��N��Ŏ�+	zؕ�!�dI���nIT�d����#��[�:��O,u;�0��
�P��-�3�>ڻ�pf���7O�Y-�A��&�q�Q#e%������B(�����Hȕș�Y�.�'c6�4᱋�:	�H���tP\c��.�QՄ�zk���9k�,!��W:~�ȈkfI����n��knN7NB�O�b�.�4ĩx���}�Ț9�"��5,:����8}\��Y{i���`_LyHHXO�m��������q��@��!h��)�a덒�4�.��Q������0��W�39�r#���^Q;Tw�A�v��yD�t4e������<���;���W�@ɢr,m���N�͗��E(<ƂӁ����B�G�ӑ]��<���"m�e����M�1/�վU��~�b�:���swt���%W;�\U!W{���y�@ǎ��v�c�}�����0��|n4H�峚e���Y�ry�<f&�=��6LK�ń��ǡc�Yguб}�Y��5b��}Ё�n$�t�ϦkS��n�J�oׄ1k3+�_���!���3�����<��v����a{�Q47���lO*2��&��q��^�{�aU W�=�>���`-P�SIZ�J-[��1���-c�qAg�ƻ�@w����v�*�#w��7:��ᦣ˾����h2�U�5eU��v.}���(�뎡c�����<H�GM��=�/�����+r����DI:hf�\��Zҡ��;��^��o}���otT@���Rt{9$ӿ����h2�UI:*��{M�/o?"&��;L����OG{ �.�f�Z����NǍr钎�s�-rU��`wcC���ڧ�����}�(u:b�ȍ���xK'��%ѽ��F�r�)�>)�8�2����0޵�dv�T��=�M,�w��X�[�����+K�7ٽ�J��냽�ܼ�(����@8Xay��20X!c���F0�W.�V�(�Ƚ��Q,�$cCS�ڇ��k�9��3��.��f���rb��f���[�K� �����[_��@�j"#��������(�.e ��n�r� c��,>�]$�@��`���Yր�]px�hi b'�.j�����<��ר�Mw�ٗ��=0�OcCE�72
�?���V���x�s��*c�]�V���Ľ`,b�18��S-ų��l�-Y���'��1�:T�dO��䧉F���bɱ�͂l�f�ۇG +�h��-�h�!�0Z�&��/ொ�]V:�@ƎB��h��f�D��*�rd��(��%˺Òe��Ī�v,�/�o�Zխ5w%�{���;5�"p-2.�tA���4��#h[���:ٛ_�������m�e�狼�i�\>��Nz�uO$ت|�[?.�NK�\����d��u"��qe8�@̽*�4�k��^u��\��r�3ӽ;������{�ڸ�)�2J�|�����qGw����F0w��/�j�$B2D
��I�F�@�gw0iGl\��(8 ;�.ĆE^5�f��EƎ���#    cǕ?�$#�RUG<�~���}ɸ�� ް׼��[��vv��	�:��ο��	ʱ:8� �/$�����S`��9�7j5���~����%VwPJ��6�{1d܃N��Ak��9�	��$#����#nmiU:ǥ��:��
A��]����0��>^�5��*2�����q���̙l���\��c ����A.��(�22r�����_gR��ݰ����2H�/GFU.�����od�9�5�q��d���&@FS,-#=n=7��F� c��	�������	NF~�c�$�`2��}ߞF����EV�m �ߵ�m���D�RM�����Q\���^S ��K�kz���tI�����ۅ|2���rd7����V�PH~iLU� C����/�f�Ɖي&Uq�.`#ǝ�G�8�T&[���,@�����କ����J�:�Ef�kAs�I'��ɋ��$�0�{�m�]$�h�eVh��W
^���,d�LbȀ��5y&4�fe�RB)[9��eэc���8b�Y�P�pP������=F�A,�h���M
n	)��+��9@f7�F��x�HGt.�mJ��9��m2�j*��dX'��m��N��D�{
�j�\���g،�i��9�*.�~:��, �A�
'���l�aG�=Tr���`z?���Q�-#�_c�X�n����d�N�J"��zOAFT��2��D�&_�q]ݗp�9�#�^�Yd��p�;��}f7,�檍	2�g���==_�>��t���?<=�z����bo,��iÐ��"d�S�12]���U�MC�xg���uE&���u�0���r� ?�<�R�x)1���s�����v|�V�Q�����O������I@#�8�R>!Y���백]�&F��`$k���8sYB��O[+�"��[7d��loj/.-����fK)ֿ>��槇�߿���>o���q��;��y�
�c��I�#r�������&�E��U�����:����+Ӳ���|�eU��H6��ϴT�9���,�/NG0�o�'ޮ��r'84����)��0*p�9T�a�$b�R�X.��b�뒨i�C����?=^�?=|��y�;��9r1�
"��:�G4`^�5m�p|�YBM	1����y�7ڜP5J��1������߰k	�Q�iĐ�D�V�Rԇ��C{��n��a�	G7'�!ٍR�@�]���P��A�I.u��rAJ�\&WZ�ɕ`W�>ΟY��>Y^0h/J���4��,�E�E!��"��.��Léݲ��D�:�_q�C�Y��e{,-m����:��@��>��iZr��M��G�\&��#Ɏl��4b�"b�9O"�f���%���3�a���������M��V�����޹nn����8���}J��uN&S�3P6>��� .f���=UQ5Q�{j�0��p�[P�zx�a�BE�3m��<2����V�G���	��rZ��KI(Y
����a�S�lkò�����x���Ⱥd�q3<�e�r�+��3W*�3��}H��F�У�$=���<��'�.w���T��`e�
��2��9�;�y�k�������,ů c��  큒�%#%��N���M���a� ��
B|�W�Nqہ�(���ڈG9F��`�4��y�lL����[9J�ֿv<y1��69vU�BaqC���GXXh��R2n0��"*H7^��̵��1w�C�.I�i�m��1q��Ae��1l���P�.�{z�������߿>|�������y;\i��j�c���������[�̑��f����```*8���t̮�� �n0#4D�l�՞��P��ApF���圴�O*1�ç�8�6�M-Y�x�"���ZI�wz��Ȭ`��᪳M&םc��M�*����䐻T1��,�p����[���X��Qq!�������ۯ"D��FݚM2��	�=��q9��������u�g���T(�QGnI-��ohiI��4f��)��&�s���[z�[G���U�٨�Z?0%!'��l�aq1	l7���9*��H��2�S�R=*�ڿo��X�L��&��hkܷb��$jq4�m$V}�d�*n����0Yʠthw
�I��������_=���K�e�Y��c?j�u��C�M�Ŵ֯D�Z��R c�՛�߾���a�ƴV�9�Ӌ���9�t&��-px�-J���܂��ES}B�`cw�^O�1l5�AvYZ��f�@�k�Λ��*���K��-���E/1$�ԩ~U�x5�6r�@����0�Ì��zK@�����ɕ/f�?�	/=�����Y�?�H� �J��*���ժ�G�`���-�uq�t�(�>��n��H^��k���(�=.O���h��4��������@ J������J0gO��w@.�c�z�٭/6��2iM8
,e��^�?�%S9񈙣W[C�#�P�LI�K���AXR����d݃����h��kg��5�GNu���Z��H�O�wU�D�j���m�F��$�d�*y9;��-	�:�r��"&�$g�5lǓ� +C���(-}�Q�Yb�y��]���8�����J:G���S5��W�[{�Nv.�`�R��9��R9���W��9��>*^�jH���`�v(�����u�Cbwƌ����e��J�ᴮ�mqR����t/�pyk��vъo�'{�u����U�N��O�괖��&q; ^1��/�ۢ*��]�u�5k���薬N��j/.2��]�B��u䰫�ꁘ�TJ��M�ǭ�=	��:9'H}���'���Y�mU�%S��w�쀘��Tǥv5l�@�ƪs��x��Q�H�Q�pZ�v���ӣg8%j ym7t+����;�?��z�Ih�.����[r�n$�`�� 9�ж�zVϬ�����+�ɩ���j��?���}7	�������[
)q6��V�8���+�{hb8U���PC�!���.��9�yv���D�JJ�?�*��}rU����"0�Ձ�'���wwH�̜���G�E��)F��	�*]���.����1�2�O����8��v��&���x����������g�pJs<fN��k�C��(I^`)k)yi��4�����
��X��|ZT��$���L?^�#��-�8~��$B8�+Z�����{}�)���g�D��|�׆-PB5�P��(��{�'ʠ�C�_⟆Lj������`ι)�*�&���lSq�"���oY��6�6r8>�1a�F�ᇷ����6C1�l!������Ue��&#�l���DO�;N>2��%�A��;�8+y��7�;�߲u�
�I�f�7]\�xv&��i�Q|ԇ��>Й	/e�����W==i�]/xj����[���8�����崍<���䍈uID�X]������Q�>3.�L:8U����Q��X�Q	�`��7d`��	�qf�'�++��jB<OW�Ȥ'��>K���?�Cv~�gN1ǖ,8���4/s�y��	�Oj����F��.$�va���\'�ܩs�����2g� �>;��Gs\��8�<l!zk$'�D�F����:��x�ڸ�?Ǣ
�4������ը6�q��b�|F)T2�$�ů��R5r�tj�f�C-��@!�B�t[���
r���0'��rk�-S�j@��,���Af�Ҟ�ͫ��疗�~�г��Rm�7fgn�~˪���ë��멎�=
��S�ZT��Ի{kf�<��]_��&���|�:SS}8Q�o��%�!q?:>̙qK����s|c^�*���o9	q�l@9^��;�v�.����zS��j��G��v��?7�O�>tsW�>���6��AE��nV$�}?���
�cC,�9,���eP`�����i����f�~ɯ�^��s�`Aɫ��1>s~�VwW��G������џ�>�S�Up�2�[�o��ħ���Ԟ�?���9�Gj����3;�{gn���SVsS*�y
OJ.���Z�*;��tl}rΉ�!����C���;MJ �  �>-ɏ��ii��}Z.Ğ5�7'۴{�*7yUUI��%H���>m*pҞ:��2��8ށ�21��b�$�Jʭ��专1��j���Xv��O���Pz��� ��m�b�i���}j"��z��r8�ޮ3��m���Eȃ��U8�pM�Co�pQiM~������W������OtH*j��7�iC�}Э����6��q<�'�}�B��`�����WL�>犓g��$8�V�n��+?�<5�2��Xu��b޺џB�|V�qvbyA�0b��o�}\���:{6n��R̟x���5���u���7�"�i����%}�"<�⮝ ����5�7���&�Ƀ�jC޾Q*a]�4����W0����� }b�nm��i��v�M�Fl;�^��7� '��V%�s�j����[��ѳ���f�Թ�<�ӣ�ͷGIs�������BK�+�w�)2�͟��)�}��I[������&���� �g�����|��P�O�}I�m����O�%7�T���_���*�u���f��ؿ�R'�lM��O�ҩ�}X�OL��ߔC߆����]r����Z+[e�1N�]U<y�tm���a�$�)!�3��:��G�Tv_��$V�����{��Г����\�G�o� �$��LN=&�$��V�����t&O}�}j��z�|9��}`�s�'�����`��S�����>�p��R�\�\��Z���ۋ!3�<��-�pr��ϱ��ʜqԫ����[`�����T8I/���%e!��-�[�k�8��>�O|����7Ċ��Xq����
.`m(+���*\��vX�:���w�a`Xo�2��F�N�p���x���|<���>(J�T���������>�ֆom��l�!�r9���!�RN��{$X6ZE4�<�2q�+���Ju)y�J�r��s��%����ꓡ_϶�uX�j�m��%��)�&����,Iz-z���ﷸ;��۵� ��at�qDA���-џx�&I�Q�4]^�p�Ύ�9~�����\�¿�A���%�6�6��&�
=��q�(�����k��O�a���i��kC��>~����燏?>]����OϿN���|�f�o���	��c)(�N#:.i��I�ӣ��ˁ��B΄3m�6��VE���k?*C�O�>�������׿ꐂv���(zۤ�p5���y�q��ο���SK&'�_@�{-k�xLצ\nm�Zs'B8��p�6�o�j�
�8�=N*ڰT��S�}��ǡ�1��
*C�X���n��>*��ćMl�Ή���*~��!�8X276��4N�sU �a"��P��؜z���џ�F���f�&��JA'��=��B�8�����'���Lm. Htp��m���t\8�C�_⟆9Ε$?���>�r�g9�x�+�ϝ\����d��7o�Lh��	��Us�t���Dl1�ۧu{�";O>�%���k��т�Z�݂�X�1��k������_��f���o0H"��׽X� �P�M6!������w�xon�ʱG!m����c/�V\F����L���ۃrx��v�g�3��� X�9��t���̯lX��G�3�X��O˜�A3� {g���,�q�Պ@Sep�^q��܀�5�
���X���d�`�0P[�?<�5g��Ė��<G&AQ�_q���H�Vk�Tۨ�`6��ލ�$n�K�	,�u��0���b��^�Þ�+����B��b��m�,~�:
t���}&����#�I�@Fm+5*�W�b��6��;>�4_=Se!�z�"럎F;U�IAiy��lj���N�©�G�SWo��RW�Y8��â w��c8 {��r��M�5Hl�f�Ä�3q`X�$�&����u��"�uH�B�����xm�U�BtS�a{�29�n�<ͦ�-���ﰬ:s�(!�)�J)�U^�^�=����ѡ@�=%���j�ea	�W��o�M�/����_��C�t����_������     